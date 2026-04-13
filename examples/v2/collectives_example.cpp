#include <functional>
#include <print>
#include <ranges>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"
#include "kamping/v2/collectives/bcast.hpp"
#include "kamping/v2/contrib/cereal_view.hpp"
#include "kamping/v2/ranges/concepts.hpp"
#include "kamping/v2/ranges/ranges.hpp"
#include "kamping/v2/tags.hpp"
#include "kamping/v2/views.hpp"

int main(int, char*[]) {
    kamping::Environment<> env;
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // ── Original: pipe chain with with_counts / with_auto_displs ─────────────
    {
        std::vector<int> v{1, 2, 3, 4};
        std::vector<int> counts{1, 2, 3};
        auto             closure = kamping::views::with_type(MPI_INT) | kamping::views::with_counts(counts);
        counts[1]                = 42;
        std::vector<int> displs;
        auto             sbuf = v | closure | kamping::views::with_auto_displs();
        static_assert(kamping::ranges::send_buffer_v<decltype(sbuf)>);
        std::println("sizev={}, displs={}", kamping::ranges::sizev(sbuf), kamping::ranges::displs(sbuf));
    }

    // ── auto_counts_view: 0-arg (owned vector, auto-resize) ──────────────────
    {
        auto counts_buf = kamping::views::auto_counts();
        static_assert(kamping::ranges::has_mpi_sizev<decltype(counts_buf)>);
        // Simulate infer(): pre-allocate and write counts.
        counts_buf.set_comm_size(3);
        counts_buf.counts()[0] = 1;
        counts_buf.counts()[1] = 2;
        counts_buf.counts()[2] = 3;
        counts_buf.commit_counts();
        std::println("auto_counts() sizev={}", kamping::ranges::sizev(counts_buf));
    }

    // ── auto_counts_view: 1-arg (user buffer, no resize) ─────────────────────
    {
        std::vector<int> user_counts(3);
        auto             counts_buf = kamping::views::auto_counts(user_counts);
        // infer writes directly into user_counts via counts_buf.counts()
        counts_buf.counts()[0] = 4;
        counts_buf.counts()[1] = 5;
        counts_buf.counts()[2] = 6;
        counts_buf.commit_counts();
        std::println("auto_counts(buf) sizev={}, user_counts={}", kamping::ranges::sizev(counts_buf), user_counts);
    }

    // ── auto_counts_view: 2-arg (user buffer + resize tag) ───────────────────
    {
        std::vector<int> user_counts; // starts empty, will be resized
        auto             counts_buf = kamping::views::auto_counts(kamping::v2::resize, user_counts);
        counts_buf.set_comm_size(3); // resizes user_counts to 3
        counts_buf.counts()[0] = 7;
        counts_buf.counts()[1] = 8;
        counts_buf.counts()[2] = 9;
        counts_buf.commit_counts();
        std::println("auto_counts(resize, buf) sizev={}", kamping::ranges::sizev(counts_buf));
    }

    // ── resize_v_view with auto displs (monotonic fast path) ─────────────────
    {
        auto counts_buf = kamping::views::auto_counts();
        counts_buf.set_comm_size(3);
        counts_buf.counts()[0] = 1;
        counts_buf.counts()[1] = 2;
        counts_buf.counts()[2] = 3;
        counts_buf.commit_counts();

        std::vector<int> recv_data;
        std::vector<int> displs_buf;
        // chain: recv_data | with_counts | with_auto_displs | resize_v
        auto rbuf = recv_data | kamping::views::with_counts(counts_buf.counts())
                    | kamping::views::with_auto_displs(kamping::v2::resize, displs_buf) | kamping::views::resize_v;
        static_assert(kamping::ranges::data_buffer_v<decltype(rbuf)>);
        // with_auto_displs always provides monotonic displs
        static_assert(kamping::ranges::has_monotonic_displs<decltype(rbuf)>);
        auto* ptr = rbuf.mpi_data(); // resizes recv_data to 1+2+3 = 6
        std::println("resize_v + auto_displs: recv_data.size()={}", recv_data.size());
        (void)ptr;
    }

    // ── resize_v_view with user displs (non-monotonic, O(p) path) ────────────
    {
        auto counts_buf = kamping::views::auto_counts();
        counts_buf.set_comm_size(3);
        counts_buf.counts()[0] = 1;
        counts_buf.counts()[1] = 2;
        counts_buf.counts()[2] = 3;
        counts_buf.commit_counts();

        std::vector<int> recv_data;
        // Non-standard displs: rank 2 at offset 10 → buffer must be 13
        std::vector<int> user_displs{0, 1, 10};
        auto             rbuf = recv_data | kamping::views::with_counts(counts_buf.counts())
                                | kamping::views::with_displs(user_displs) | kamping::views::resize_v;
        static_assert(kamping::ranges::data_buffer_v<decltype(rbuf)>);
        static_assert(!kamping::ranges::has_monotonic_displs<decltype(rbuf)>);
        auto* ptr = rbuf.mpi_data(); // max(0+1, 1+2, 10+3) = 13
        std::println("resize_v + user_displs: recv_data.size()={}", recv_data.size());
        (void)ptr;
    }

    // ── resize_v_view with user-declared monotonic displs (fast path) ─────────
    {
        auto counts_buf = kamping::views::auto_counts();
        counts_buf.set_comm_size(3);
        counts_buf.counts()[0] = 1;
        counts_buf.counts()[1] = 2;
        counts_buf.counts()[2] = 3;
        counts_buf.commit_counts();

        std::vector<int> recv_data;
        std::vector<int> mono_displs{0, 1, 3}; // exclusive scan of {1,2,3}
        auto rbuf = recv_data | kamping::views::with_counts(counts_buf.counts())
                    | kamping::views::with_displs(kamping::v2::monotonic, mono_displs) | kamping::views::resize_v;
        static_assert(kamping::ranges::data_buffer_v<decltype(rbuf)>);
        static_assert(kamping::ranges::has_monotonic_displs<decltype(rbuf)>);
        auto* ptr = rbuf.mpi_data(); // fast path: displs[2]+counts[2] = 3+3 = 6
        std::println("resize_v + monotonic user_displs: recv_data.size()={}", recv_data.size());
        (void)ptr;
    }

    // ── commit_counts() propagates through chain and invalidates displs ────────
    // Use a plain std::vector so that all() wraps it in ref_view (borrow, not copy),
    // making count changes visible through the chain after commit_counts().
    {
        std::vector<int> counts_vec{1, 2, 3};
        std::vector<int> recv_data;
        std::vector<int> displs_buf;
        auto             chain = recv_data | kamping::views::with_counts(counts_vec)
                                 | kamping::views::with_auto_displs(kamping::v2::resize, displs_buf);

        // First displs computation
        std::println("displs before commit: {}", kamping::ranges::displs(chain));

        // Simulate infer() on a second call: counts changed
        counts_vec[0] = 10;
        counts_vec[1] = 10;
        counts_vec[2] = 10;
        chain.commit_counts(); // propagates to with_auto_displs_view, invalidates cached displs

        // Displs recomputed from new counts
        std::println("displs after commit: {}", kamping::ranges::displs(chain));
    }

    // ── with_counts: lvalue ref borrows — mutation to original is visible ────────
    {
        std::vector<int> data{1, 2, 3, 4, 5, 6};
        std::vector<int> cv{1, 2, 3};
        auto             view = data | kamping::views::with_counts(cv);
        cv[0]                 = 99; // mutate original after constructing the view
        std::println("with_counts lvalue borrow: sizev={}", kamping::ranges::sizev(view));
        // → [99, 2, 3]
    }

    // ── with_counts: rvalue owns — extract the vector back out ────────────────────
    {
        std::vector<int> data{1, 2, 3, 4, 5, 6};
        auto             view = data | kamping::views::with_counts(std::vector<int>{1, 2, 3});
        auto             cv   = std::move(view).counts().base(); // move out of owning_view
        std::println("with_counts rvalue extract: counts={}", cv);
        // → [1, 2, 3]
    }

    // ── with_displs: lvalue ref borrows — mutation to original is visible ─────────
    {
        std::vector<int> data{1, 2, 3, 4, 5, 6};
        std::vector<int> cv{1, 2, 3};
        std::vector<int> dv{0, 1, 3};
        auto             view = data | kamping::views::with_counts(cv) | kamping::views::with_displs(dv);
        dv[2]                 = 10; // mutate original
        std::println("with_displs lvalue borrow: displs={}", kamping::ranges::displs(view));
        // → [0, 1, 10]
    }

    // ── with_displs: rvalue owns — extract the vector back out ───────────────────
    {
        std::vector<int> data{1, 2, 3, 4, 5, 6};
        std::vector<int> cv{1, 2, 3};
        auto view = data | kamping::views::with_counts(cv) | kamping::views::with_displs(std::vector<int>{0, 1, 3});
        auto dv   = std::move(view).displs().base();
        std::println("with_displs rvalue extract: displs={}", dv);
        // → [0, 1, 3]
    }

    // ── with_auto_displs: lvalue container borrows — computed displs written to original
    {
        std::vector<int> data{1, 2, 3, 4, 5, 6};
        std::vector<int> cv{1, 2, 3};
        std::vector<int> dv(3); // pre-sized; all() borrows via ref_view
        auto             view = data | kamping::views::with_counts(cv) | kamping::views::with_auto_displs(dv);
        (void)kamping::ranges::displs(view); // triggers exclusive_scan into dv
        std::println("with_auto_displs lvalue: displs={}, original dv={}", kamping::ranges::displs(view), dv);
        // → [0, 1, 3], dv = [0, 1, 3]
    }

    // ── with_auto_displs: rvalue container owns — extract computed displs back out
    {
        std::vector<int> data{1, 2, 3, 4, 5, 6};
        std::vector<int> cv{1, 2, 3};
        auto             view = data | kamping::views::with_counts(cv)
                                | kamping::views::with_auto_displs(kamping::v2::resize, std::vector<int>{});
        (void)kamping::ranges::displs(view); // resizes and fills via exclusive_scan
        auto dv = std::move(view).displs().base();
        std::println("with_auto_displs rvalue extract: displs={}", dv);
        // → [0, 1, 3]
    }

    // ── auto_counts owned: counts() is non-copyable lvalue → all() borrows via ref_view
    {
        auto counts_buf = kamping::views::auto_counts();
        counts_buf.set_comm_size(3);
        counts_buf.counts()[0] = 1;
        counts_buf.counts()[1] = 2;
        counts_buf.counts()[2] = 3;
        counts_buf.commit_counts();
        // counts_buf.counts() returns owning_view<vector<int>>& (non-copyable kamping lvalue)
        // → all() wraps in ref_view<owning_view<...>>; same as passing a plain lvalue range.
        std::vector<int> data;
        auto rbuf = data | kamping::views::with_counts(counts_buf.counts()) | kamping::views::with_auto_displs()
                    | kamping::views::resize_v;
        (void)rbuf.mpi_data(); // resizes data to 1+2+3 = 6
        std::println("auto_counts owned + with_counts(counts()): data.size()={}", data.size());
        // Extract the owned vector back out
        auto vec = std::move(counts_buf).counts().base();
        std::println("auto_counts owned extract: counts={}", vec);
        // → [1, 2, 3]
    }

    // ── auto_counts user lvalue: counts() is copyable ref_view → passes through as copy
    {
        std::vector<int> cv(3);
        auto             counts_buf = kamping::views::auto_counts(cv);
        counts_buf.counts()[0]      = 4;
        counts_buf.counts()[1]      = 5;
        counts_buf.counts()[2]      = 6;
        // counts_buf.counts() returns ref_view<vector<int>>& (copyable) → copies the pointer
        std::vector<int> data;
        auto rbuf = data | kamping::views::with_counts(counts_buf.counts()) | kamping::views::with_auto_displs()
                    | kamping::views::resize_v;
        (void)rbuf.mpi_data(); // resizes data to 4+5+6 = 15
        std::println("auto_counts lvalue + with_counts(counts()): data.size()={}", data.size());
        std::println("auto_counts lvalue original cv={}", cv); // writes through to cv
        // → cv = [4, 5, 6]
    }

    {
        int val = 0;
        if (kamping::world_rank() == 0) {
            val = 42;
        }
        auto result = kamping::v2::bcast(val | kamping::views::serialize);
        std::println("[R{}] bcast result={}", kamping::world_rank(), *result);
    }

    return 0;
}
