# v2 TODO

## Handle types

- [x] **`mpi::experimental::status`** / **`status_view`** — done (`include/mpi/status.hpp`); re-exported from `kamping::v2::` via `include/kamping/v2/status.hpp`
- [x] **`mpi::experimental::comm_view`** — non-owning wrapper (`include/mpi/comm.hpp`); CRTP mixin `comm_accessors` with `.rank()`, `.size()`, `.native()`
- [ ] **`kamping::v2::comm`** — owning wrapper; calls `MPI_Comm_free` on destruction (for subcommunicators)
- [ ] **`kamping::v2::request_view`** — non-owning wrapper over `MPI_Request*`
  - Satisfies `bridge::convertible_to_mpi_handle_ptr<MPI_Request>`
  - Counterpart to `iresult` for interop with external request arrays
  - No owning `request` planned — `iresult` covers the RAII case

## P2P

- [x] **`mpi::experimental::send`** / **`v2::send`** (all modes: standard, buffered, sync, ready)
- [x] **`mpi::experimental::recv`** / **`v2::recv`**
- [x] **`mpi::experimental::isend`** / **`v2::isend`**
- [x] **`mpi::experimental::irecv`** / **`v2::irecv`**
- [x] **`mpi::experimental::sendrecv`** / **`v2::sendrecv`**
- [x] **`mpi::experimental::isendrecv`** / **`v2::isendrecv`**
- [x] **`mpi::experimental::probe`** — bare `MPI_Probe` wrapper (core only)
- [x] **`mpi::experimental::mprobe`** — bare `MPI_Mprobe` wrapper (core only)
- [x] **`mpi::experimental::mrecv`** / **`mpi::experimental::imrecv`**
- [ ] **`probe_result` type** (`p2p/probe_result.hpp`)
  - Owns `MPI_Message` and `MPI_Status` from a matched probe
  - Accessors: `.source()`, `.tag()`, `.count<T>()`
  - `.mrecv(rbuf)` — blocking matched receive; resizes buffer from known count
  - `.imrecv(rbuf)` — non-blocking matched receive; returns `iresult<RBuf>`
- [ ] **`v2::mprobe`** / **`v2::improbe`** (`p2p/mprobe.hpp`)
  - `mprobe(source, tag, comm)` → `probe_result`
  - `improbe(source, tag, comm)` → `std::optional<probe_result>`
  - Update `infer(comm_op::recv, ...)` to use `probe_result` instead of raw `MPI_Message`

## Clean layer split + file restructuring

**Decision: views-free core.** The `core::` layer contains only the buffer contract (concepts,
accessor dispatch, `mpi_span`/`mpi_span_v`) and the bare MPI wrappers. All view machinery
(`view_interface`, `ref_view`, `owning_view`, `all()`, adaptor infrastructure, `with_*` views)
belongs exclusively to the language-bindings layer and must not appear in any header that
`core::` functions need to include.

The `core::` function signatures do **not** change — they remain concept-constrained templates
accepting anything satisfying `send_buffer` / `recv_buffer` / `send_buffer_v` / `recv_buffer_v`.
`mpi_span` and `mpi_span_v` are concrete minimal implementations of those concepts for callers
who prefer not to use the view pipeline.

**Core files move to `include/mpi/` now** (before the monorepo restructure). This establishes
the physical boundary immediately and makes the later monorepo step a trivial `git mv` with no
logic changes. The intermediate layout (actual filenames differ slightly from the plan):

```
include/
  mpi/                         ← core layer (views-free, self-contained) ✓ DONE
    collectives/
      allgather.hpp            (mpi::experimental:: namespace) ✓
      allgatherv.hpp           ✓
      alltoall.hpp             ✓
      alltoallv.hpp            ✓
      bcast.hpp                ✓
      barrier.hpp              ✓
    p2p/
      send.hpp                 ✓
      recv.hpp                 ✓
      isend.hpp                ✓
      irecv.hpp                ✓
      sendrecv.hpp             ✓
      isendrecv.hpp            ✓
      probe.hpp                ✓
      mprobe.hpp               ✓
      mrecv.hpp                ✓
      imrecv.hpp               ✓
    buffer.hpp                 (buffer concepts + accessor dispatch; was concepts.hpp+ranges.hpp) ✓
    mpi_span.hpp               (mpi_span / mpi_span_v) ← TODO Step 2
    handle.hpp                 (was native_handle.hpp) ✓
    error.hpp                  (was error_handling.hpp) ✓
  kamping/v2/                  ← ergonomics layer; #includes from <mpi/...>
    collectives/               (kamping::v2:: only; one-liners calling infer + mpi::experimental::)
    p2p/
    views/
    infer.hpp
    result.hpp
    iresult.hpp
    ...
  kamping/                     ← v1 untouched
```

### Namespace alignment

**Decision:** core layer moves to `mpi::experimental::`, ergonomics layer stays `kamping::v2::`.

| Was | Now (done) |
|---|---|
| `kamping::ranges::` (concepts, accessor dispatch) | `mpi::experimental::` ✓ |
| `kamping::core::` (bare MPI wrappers) | `mpi::experimental::` ✓ |
| `kamping::views::` (view adaptors) | `kamping::views::` ✓ |
| `kamping::v2::` (high-level wrappers, infer) | `kamping::v2::` ✓ |

### Step-by-step tasks

**[x] Step 1 — Move view infrastructure out of `ranges/` into `views/`**

  | File | Action |
  |---|---|
  | `ranges/view_interface.hpp` | `git mv` → `views/view_interface.hpp` ✓ |
  | `ranges/all.hpp` | `git mv` → `views/all.hpp` ✓ |
  | `ranges/adaptor.hpp` | `git mv` → `views/adaptor.hpp` ✓ |
  | `ranges/concepts.hpp` | stays ✓ |
  | `ranges/ranges.hpp` | stays ✓ |

  `ranges/` now contains only headers that the core layer legitimately needs.

- [x] **Step 2 — Add `mpi_span` / `mpi_span_v`** (`include/mpi/mpi_span.hpp`) ✓

  Concrete non-template structs satisfying the buffer concepts without any view machinery.
  `void*` covers both send (`void const*` is implicit) and recv:

  ```cpp
  struct mpi_span {
      void*          ptr;
      std::ptrdiff_t size;
      MPI_Datatype   type;

      void*          mpi_ptr()  noexcept       { return ptr; }
      std::ptrdiff_t mpi_count()  const noexcept { return size; }
      MPI_Datatype   mpi_type()  const noexcept { return type; }
  };

  struct mpi_span_v {
      void*        ptr;
      MPI_Datatype type;
      int const*   counts;      // per-rank element counts (length: comm_size)
      int const*   displs;      // per-rank displacements  (length: comm_size)
      int          comm_size;

      void*                mpi_ptr()   noexcept       { return ptr; }
      MPI_Datatype         mpi_type()   const noexcept { return type; }
      std::span<int const> mpi_counts() const noexcept { return {counts, static_cast<std::size_t>(comm_size)}; }
      std::span<int const> mpi_displs() const noexcept { return {displs, static_cast<std::size_t>(comm_size)}; }
  };
  ```

  - `mpi_span` satisfies `send_buffer` and `recv_buffer`
  - `mpi_span_v` satisfies `send_buffer_v` and `recv_buffer_v` (no scalar `mpi_count()`)

**[x] Step 3 — Split collectives + p2p files and move core halves to `include/mpi/`**

  Done. All collectives and p2p files are split; core halves live in `include/mpi/`.
  Infrastructure files placed as follows (final names differ from plan):
  - `ranges/concepts.hpp` + `ranges/ranges.hpp` → `include/mpi/buffer.hpp` (merged)
  - `native_handle.hpp` → `include/mpi/handle.hpp`
  - `error_handling.hpp` → `include/mpi/error.hpp`
  - CMakeLists updated to add `include/mpi` to include path.

**[x] Step 4 — Namespace rename**

  Done in the same pass as Step 3. All `include/mpi/` code is already in `mpi::experimental::`.
  All `include/kamping/v2/` references updated accordingly.

- [x] **Step 5 — Verify include discipline**

  Verified: no file under `include/mpi/` includes anything from `include/kamping/v2/views/`,
  `infer.hpp`, or result headers.

## Monorepo restructure

**Prerequisite: layer split + file restructuring must be complete** — `include/mpi/` must be
self-contained and validated before this step.

After step 5 above, the monorepo restructure is purely mechanical:

Proposed top-level layout:

```
/
  mpi-core/         include/mpi/ moves here — extractable via git subtree split
  kamping-v2/       include/kamping/v2/ moves here
  kamping-v1/       existing v1 code (while it lives); include/kamping/ paths unchanged
  CMakeLists.txt    root; adds all subdirs, wires inter-component CMake targets
```

### Tasks

- [ ] **Move files** using `git mv` to preserve history:
  - `include/mpi/` → `mpi-core/include/mpi/`
  - `include/kamping/v2/` → `kamping-v2/include/kamping/v2/`
  - `include/kamping/` → `kamping-v1/include/kamping/` (if v1 survives)

- [ ] **Per-component `CMakeLists.txt`** with explicit `target_link_libraries` edges
  (`kamping-v2` → `mpi-core`, `mpi-core` → MPI + kamping-types).

- [ ] **Root `CMakeLists.txt`** wires components together; shared CI and docs remain
  at the repo root — single pipeline covers everything.

- [ ] **Update FetchContent / find_package** docs. v1 users change only the `SOURCE_SUBDIR`
  in FetchContent; their `#include <kamping/...>` paths are unaffected.

Include paths and namespace are settled: headers under `include/mpi/` (no `experimental/`
subdirectory), namespace `mpi::experimental::`. When standardized: one namespace rename,
no file moves. kamping-v2 stays `kamping::v2::` throughout.

## Reduction operation handling

Reduction collectives (`reduce`, `allreduce`, `scan`, `exscan`) need to resolve a user-provided
op argument to an `MPI_Op` at the point of the MPI call. The design mirrors the buffer accessor
system (`kamping::core::type/size/data`) so op resolution is externalized rather than baked into
each collective's implementation.

### `builtin_mpi_handle` — add `MPI_Op`

- [ ] Add `MPI_Op` to the `builtin_mpi_handle` concept in `handle.hpp` so that raw
  `MPI_Op` values pass through `mpi::experimental::handle` unchanged, consistent with `MPI_Comm`,
  `MPI_Datatype`, etc.

### `native_handle_traits` for kamping-types op/type wrappers

- [ ] Specialize `mpi::experimental::handle_traits` for `kamping::types::ScopedOp`,
  `kamping::types::ScopedFunctorOp`, and `kamping::types::ScopedCallbackOp` so they satisfy
  `convertible_to_mpi_handle<MPI_Op>` and can be passed directly to `mpi::experimental::` collectives.
- [ ] Specialize `mpi::experimental::handle_traits` for `kamping::types::ScopedDatatype` so it
  satisfies `convertible_to_mpi_handle<MPI_Datatype>` — the only kamping-types datatype wrapper
  that needs to reach the core layer (committed derived types; complex type pools belong in v2).

### `mpi::experimental::op_traits<Op, SBuf>` — buffer-aware customization point

`kamping-types` keeps `mpi_operation_traits<Op, T>` element-type-centric (standalone module,
no buffer concept dependency). The core layer owns a separate buffer-aware trait:

```cpp
// primary — not a builtin by default
template <typename Op, data_buffer SBuf>
struct op_traits {
    static constexpr bool is_builtin = false;
};

// default for ranges — delegates to kamping-types
template <typename Op, std::ranges::range SBuf>
    requires types::mpi_operation_traits<Op, std::ranges::range_value_t<SBuf>>::is_builtin
struct op_traits<Op, SBuf> {
    static constexpr bool is_builtin = true;
    static MPI_Op op() {
        return types::mpi_operation_traits<Op, std::ranges::range_value_t<SBuf>>::op();
    }
};
```

Users can specialize `mpi::experimental::op_traits<Op, SBuf>` non-intrusively for custom buffer+op
combinations, parallel to `mpi::experimental::buffer_traits<T>` for buffer types.

### `mpi_op_for<Op, SBuf>` concept

```cpp
template <typename Op, typename SBuf>
concept mpi_op_for = bridge::convertible_to_mpi_handle<Op, MPI_Op>
                  || (std::ranges::range<SBuf> && op_traits<Op, SBuf>::is_builtin);
```

The `std::ranges::range<SBuf>` guard is required because `op_traits`'s default range
specialization accesses `range_value_t<SBuf>`, which is only valid for range types.
For raw-pointer / `mpi_span` buffers only `convertible_to_mpi_handle<MPI_Op>` applies —
builtin functor inference requires a typed range.

### `mpi::experimental::native_op<SBuf>(op)` — free function customization point

Analogous to `mpi::experimental::type(buf)` / `mpi::experimental::count(buf)`:

```cpp
template <data_buffer SBuf, typename Op>
    requires mpi_op_for<Op, SBuf>
MPI_Op native_op(Op const& op) {
    if constexpr (mpi::experimental::convertible_to_handle<Op, MPI_Op>)
        return mpi::experimental::handle(op);
    else
        return op_traits<Op, SBuf>::op();
}
```

`T` is never extracted at the top of `native_op` — `op_traits<Op, SBuf>::op()` is only
instantiated in the `else` branch, where `mpi_op_for` already guarantees `SBuf` is a range.

### Inplace handling for reduction collectives

Reduction collectives (reduce, allreduce, scan, exscan) are fundamentally asymmetric across ranks:
the buffer role differs depending on whether it's inplace and on which rank the operation is.

**Design decision:** Branch on `ptr(sbuf) == MPI_IN_PLACE` to determine the correct interpretation:
- **Inplace** (`sbuf` is sentinel): count/type from `rbuf`; only valid on root
- **Normal** (two-buffer): count/type from `sbuf`; on root, must match `rbuf` (asserted)

This avoids forcing reduce into a false "recv-centric" model where the receive buffer describes
all ranks' contributions.

```cpp
template <send_buffer SBuf, recv_buffer RBuf, typename Op>
void reduce(SBuf&& sbuf, RBuf&& rbuf, Op&& op, int root, MPI_Comm comm) {
    auto sbuf_ptr = ptr(sbuf);
    int rank = 0;
    MPI_Comm_rank(handle(comm), &rank);
    int root_rank = to_rank(root);

    if (sbuf_ptr == MPI_IN_PLACE) {
        // Inplace: count and type from rbuf
        KASSERT(rank == root_rank, "inplace reduce only valid on root");
        MPI_Reduce(sbuf_ptr, ptr(rbuf), count(rbuf), type(rbuf), as_mpi_op(op, sbuf, rbuf), root_rank, comm);
    } else {
        // Normal: count and type from sbuf
        KASSERT(rank != root_rank || count(sbuf) == count(rbuf), "on root: counts must match");
        KASSERT(rank != root_rank || type(sbuf) == type(rbuf), "on root: types must match");
        MPI_Reduce(sbuf_ptr, ptr(rbuf), count(sbuf), type(sbuf), as_mpi_op(op, sbuf, rbuf), root_rank, comm);
    }
}
```

The infer() protocol remains unchanged: it only sets recv counts on root, and deferred buffers on
non-root are simply never given the chance to be deferred (they are null_buf or similar).

The same pattern applies to `allreduce`, `scan`, `exscan`.

### v2 responsibility

Custom functor → `MPI_Op` creation (i.e. `ScopedFunctorOp`, `ScopedCallbackOp`) is the
caller's responsibility at the core level. v2 may provide convenience wrappers that
create and lifetime-manage these objects before delegating to `core::`.

## Sentinels (`include/kamping/v2/tags.hpp`)

Three zero-overhead sentinel buffer types. All implemented.

- [x] **`v2::inplace`** (`inplace_t`) — passes `MPI_IN_PLACE` / 0 / `MPI_DATATYPE_NULL`; satisfies
  `send_buffer` and `recv_buffer`. Used as the send argument for inplace collective overloads
  and available for explicit use in two-buffer forms.
- [x] **`v2::null_buf`** (`null_buf_t`) — passes `nullptr` / 0 / `MPI_DATATYPE_NULL`; satisfies
  `send_buffer` and `recv_buffer`. Used internally by non-root shorthand overloads; also
  exposed for users who want explicit uniform call sites across root and non-root.
- [x] **`v2::bottom`** (`bottom_t`) — passes `MPI_BOTTOM` only; exposes `mpi_data()` but **not**
  `mpi_size()` or `mpi_type()`. Not a `data_buffer` on its own — must be composed with
  `views::with_type` and `views::with_size` before use. The existing `view_interface`
  conditional forwarding already handles partial bases correctly; no view changes needed.

  ```cpp
  v2::send(v2::bottom | views::with_type(my_abs_type) | views::with_size(1), dest, comm);
  ```

## `recv_v<T>()` helper

- [x] **`v2::auto_recv_v<T, Cont = std::vector<T>>()`** / **`views::auto_recv_v`** — pipe adaptor and
  owned factory for the common variadic recv buffer idiom. The adaptor composes
  `auto_counts() | auto_displs() | resize_v`; the factory wraps an owned `Cont{}`:

  ```cpp
  v2::allgatherv(local_data, v2::auto_recv_v<int>(), comm);
  recv_buf | views::auto_recv_v   // lvalue borrow form
  ```

## Collectives

- [x] **`mpi::experimental::barrier`** / **`v2::barrier`**
- [x] **`mpi::experimental::bcast`** / **`v2::bcast`**
- [x] **`mpi::experimental::allgather`** / **`v2::allgather`**
- [x] **`mpi::experimental::allgatherv`** / **`v2::allgatherv`**
- [x] **`mpi::experimental::alltoall`** / **`v2::alltoall`**
- [x] **`mpi::experimental::alltoallv`** / **`v2::alltoallv`**
- [x] **`mpi::experimental::reduce`** / **`v2::reduce`** (core + v2 + infer); demonstrates inplace handling pattern for all reduction collectives
- [x] **`mpi::experimental::allreduce`** / **`v2::allreduce`** (core + v2 + infer); symmetric, no root
- [ ] **Symmetric reduction** (defer): `scan`, `exscan` — follow reduce/allreduce inplace pattern
- [x] **`mpi::experimental::gather`** / **`v2::gather`** (core + v2 + infer); demonstrates asymmetric collective pattern
- [x] **`mpi::experimental::gatherv`** / **`v2::gatherv`** (core + v2 + infer); introduces `null_buf_v` (core) and `auto_null_recv_v()` (v2) for non-root deferred participation
- [x] **`mpi::experimental::scatter`** / **`v2::scatter`** (core + v2 + infer); two-buffer + non-root shorthand; inplace on root via MPI_IN_PLACE; infer uses MPI_Bcast of per-rank count
- [x] **`mpi::experimental::scatterv`** / **`v2::scatterv`** (core + v2 + infer); no shorthand (non-root passes `null_buf_v`); infer uses MPI_Scatter of sendcounts
- [ ] **Non-blocking** (defer): all `i*` variants — leverage iresult infrastructure from p2p
- Architecture proven: `core::` wraps MPI directly, `v2::` handles inference and result types

### Collective interface design

#### Inplace single-buffer overloads (symmetric collectives only)

Single-buffer = inplace, but **only** for collectives where all ranks have the same
relationship to the buffer. For asymmetric collectives (gather, scatter, reduce) the buffer
plays different roles on root vs non-root (different semantics, different sizes) — no
single-buffer form there.

| Collective | Single-buffer form | Notes |
|---|---|---|
| `allreduce(buf, op, comm)` | ✗ | Rejected: implicit inplace hurts readability; `allreduce(inplace, buf, op)` is already clean |
| `scan(buf, op, comm)` | ✗ | Same reasoning as allreduce |
| `exscan(buf, op, comm)` | ✗ | Same reasoning as allreduce |
| `allgather(buf, comm)` | ✓ | All ranks: local data already at `rank*n` slot, rest filled in |
| `gather` / `scatter` / `reduce` | ✗ | Buffer role differs by rank — always two-buffer |

The inplace allgather precondition (local data pre-placed at `rank * n`) is implicit and
documented but not enforced.

**Decision (allreduce/scan/exscan):** No implicit single-buffer inplace form. The asymmetry with
`reduce(sbuf, op, root)` (which means non-root send-only, not inplace) would make the rule
"single-buffer = inplace" inconsistent across collectives. Prefer the explicit
`allreduce(v2::inplace, buf, op)` which is already clean compared to the raw MPI C API.

#### Asymmetric collectives: gather / scatter / reduce

Three overload tiers — all ranks always call `infer()` regardless of which overload is used,
since `infer()` may itself be collective (e.g. `MPI_Bcast` inside scatter's infer to
distribute the per-rank count). Branching logic lives inside `infer()`.

```
gather(sbuf, root, comm)              // non-root shorthand — null_buf internally
gather(sbuf, rbuf, root, comm)        // full two-buffer; null_buf on non-root, real rbuf on root

scatter(rbuf, root, comm)             // non-root shorthand
scatter(sbuf, rbuf, root, comm)       // full two-buffer

reduce(sbuf, op, root, comm)          // non-root shorthand
reduce(sbuf, rbuf, op, root, comm)    // full two-buffer
```

`v2::null_buf` is exposed and usable in the two-buffer form for users who prefer uniform
call sites (root and non-root both call the 4-arg form, non-root passes `null_buf`).

**Rejected alternatives:**
- `std::optional<RBuf>` overload: does not eliminate the rank-check branch (moves it to
  construction site); deferred resize inside an optional is unspellable; extra `infer`
  overloads for every collective. Not worth the complexity.
- Single-buffer inplace for gather/scatter/reduce: buffer means recv on root, send on
  non-root — inconsistent role, inconsistent size. Rejected for asymmetric collectives.
