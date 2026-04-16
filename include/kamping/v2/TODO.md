# v2 TODO

## Handle types

- [x] **`kamping::v2::status`** / **`status_view`** — done (`status.hpp`)
- [ ] **`kamping::v2::comm`** / **`comm_view`** — owning/non-owning wrappers around `MPI_Comm`
  - CRTP mixin `comm_accessors` with `.rank()`, `.size()`, `.native()`
  - `comm_view`: non-owning (e.g. for `MPI_COMM_WORLD`); satisfies `bridge::convertible_to_mpi_handle<MPI_Comm>`
  - `comm`: owning; calls `MPI_Comm_free` on destruction (for subcommunicators)
- [ ] **`kamping::v2::request_view`** — non-owning wrapper over `MPI_Request*`
  - Satisfies `bridge::convertible_to_mpi_handle_ptr<MPI_Request>`
  - Counterpart to `iresult` for interop with external request arrays
  - No owning `request` planned — `iresult` covers the RAII case

## P2P

- [x] **`core::send`** / **`v2::send`** (all modes: standard, buffered, sync, ready)
- [x] **`core::recv`** / **`v2::recv`**
- [x] **`core::isend`** / **`v2::isend`**
- [x] **`core::irecv`** / **`v2::irecv`**
- [x] **`core::sendrecv`** / **`v2::sendrecv`**
- [x] **`core::isendrecv`** / **`v2::isendrecv`**
- [x] **`core::probe`** — bare `MPI_Probe` wrapper (core only)
- [x] **`core::mprobe`** — bare `MPI_Mprobe` wrapper (core only)
- [x] **`core::mrecv`** / **`core::imrecv`**
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
logic changes. The intermediate layout:

```
include/
  mpi/                         ← core layer (views-free, self-contained)
    collectives/
      allgather.hpp            (mpi::experimental:: namespace)
      allgatherv.hpp
      alltoall.hpp
      alltoallv.hpp
      bcast.hpp
      barrier.hpp
    p2p/
      send.hpp
      recv.hpp
      isend.hpp
      irecv.hpp
      sendrecv.hpp
      isendrecv.hpp
      probe.hpp
      mprobe.hpp
      mrecv.hpp
      imrecv.hpp
    concepts.hpp               (buffer concepts, deferred protocol)
    ranges.hpp                 (accessor dispatch, buffer_traits)
    mpi_span.hpp               (mpi_span / mpi_span_v)
    native_handle.hpp
    error_handling.hpp
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

| Current | Final |
|---|---|
| `kamping::ranges::` (concepts, accessor dispatch) | `mpi::experimental::` |
| `kamping::core::` (bare MPI wrappers) | `mpi::experimental::` |
| `kamping::views::` (view adaptors) | `kamping::views::` |
| `kamping::v2::` (high-level wrappers, infer) | `kamping::v2::` |

The namespace rename is done as a **separate commit** from the file moves so each step can
be built and tested independently.

### Step-by-step tasks

**Step 1 — Move view infrastructure out of `ranges/` into `views/`** (by hand)

  | File | Action |
  |---|---|
  | `ranges/view_interface.hpp` | `git mv` → `views/view_interface.hpp` |
  | `ranges/all.hpp` | `git mv` → `views/all.hpp` |
  | `ranges/adaptor.hpp` | `git mv` → `views/adaptor.hpp` |
  | `ranges/concepts.hpp` | stays |
  | `ranges/ranges.hpp` | stays |

  Update all `#include` paths in `views/` files, `iresult.hpp`, `result.hpp`, and `views.hpp`.
  After this step `ranges/` contains only headers that `core::` legitimately needs.

- [ ] **Step 2 — Add `mpi_span` / `mpi_span_v`** (`include/kamping/v2/ranges/mpi_span.hpp`) ← Claude writes this

  Concrete non-template structs satisfying the buffer concepts without any view machinery.
  `void*` covers both send (`void const*` is implicit) and recv:

  ```cpp
  struct mpi_span {
      void*          data;
      std::ptrdiff_t size;
      MPI_Datatype   type;

      void*          mpi_data()  noexcept       { return data; }
      std::ptrdiff_t mpi_size()  const noexcept { return size; }
      MPI_Datatype   mpi_type()  const noexcept { return type; }
  };

  struct mpi_span_v {
      void*          data;
      MPI_Datatype   type;
      int const*     counts;      // per-rank element counts (length: comm_size)
      int const*     displs;      // per-rank displacements  (length: comm_size)
      int            comm_size;

      void*                mpi_data()   noexcept       { return data; }
      MPI_Datatype         mpi_type()   const noexcept { return type; }
      std::ptrdiff_t       mpi_size()   const noexcept {
          return std::accumulate(counts, counts + comm_size, std::ptrdiff_t{0});
      }
      std::span<int const> mpi_sizev()  const noexcept { return {counts, static_cast<std::size_t>(comm_size)}; }
      std::span<int const> mpi_displs() const noexcept { return {displs, static_cast<std::size_t>(comm_size)}; }
  };
  ```

  - `mpi_span` satisfies `send_buffer` and `recv_buffer`
  - `mpi_span_v` satisfies `send_buffer_v` and `recv_buffer_v`

**Step 3 — Split collectives + p2p files and move core halves to `include/mpi/`** (by hand)

  Each file currently contains both `kamping::core::` and `kamping::v2::` in the same `.hpp`.
  Split each file:
  - Core half → `include/mpi/collectives/<name>.hpp` (keep `kamping::core::` namespace for now)
  - v2 half → stays in `include/kamping/v2/collectives/<name>.hpp`, `#include`s from `<mpi/collectives/<name>.hpp>`

  Same split for all `p2p/` files. Also move:
  - `ranges/concepts.hpp` → `include/mpi/concepts.hpp`
  - `ranges/ranges.hpp` → `include/mpi/ranges.hpp`
  - `ranges/mpi_span.hpp` → `include/mpi/mpi_span.hpp`
  - `native_handle.hpp` → `include/mpi/native_handle.hpp`
  - `error_handling.hpp` → `include/mpi/error_handling.hpp`

  Update CMakeLists to add `include/mpi` to the include path. Build and test.

**Step 4 — Namespace rename** (separate commit, after step 3 builds cleanly)

  - `kamping::ranges::` → `mpi::experimental::` throughout `include/mpi/`
  - `kamping::core::` → `mpi::experimental::` throughout `include/mpi/`
  - Update all references in `include/kamping/v2/` that call into the core layer
  - Build and test again

- [ ] **Step 5 — Verify include discipline**

  No file under `include/mpi/` should include anything from `include/kamping/v2/views/`:
  ```bash
  grep -r "kamping/v2/views\|kamping/v2/infer\|kamping/v2/result" include/mpi/
  ```
  Should return nothing.

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

- [ ] Add `MPI_Op` to the `builtin_mpi_handle` concept in `native_handle.hpp` so that raw
  `MPI_Op` values pass through `bridge::native_handle` unchanged, consistent with `MPI_Comm`,
  `MPI_Datatype`, etc.

### `native_handle_traits` for kamping-types op/type wrappers

- [ ] Specialize `bridge::native_handle_traits` for `kamping::types::ScopedOp`,
  `kamping::types::ScopedFunctorOp`, and `kamping::types::ScopedCallbackOp` so they satisfy
  `convertible_to_mpi_handle<MPI_Op>` and can be passed directly to `core::` collectives.
- [ ] Specialize `bridge::native_handle_traits` for `kamping::types::ScopedDatatype` so it
  satisfies `convertible_to_mpi_handle<MPI_Datatype>` — the only kamping-types datatype wrapper
  that needs to reach the core layer (committed derived types; complex type pools belong in v2).

### `kamping::core::op_traits<Op, SBuf>` — buffer-aware customization point

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

Users can specialize `kamping::core::op_traits<Op, SBuf>` non-intrusively for custom buffer+op
combinations, parallel to `kamping::ranges::buffer_traits<T>` for buffer types.

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

### `kamping::core::native_op<SBuf>(op)` — free function customization point

Analogous to `kamping::core::type(buf)` / `kamping::ranges::size(buf)`:

```cpp
template <data_buffer SBuf, typename Op>
    requires mpi_op_for<Op, SBuf>
MPI_Op native_op(Op const& op) {
    if constexpr (bridge::convertible_to_mpi_handle<Op, MPI_Op>)
        return bridge::native_handle(op);
    else
        return op_traits<Op, SBuf>::op();
}
```

`T` is never extracted at the top of `native_op` — `op_traits<Op, SBuf>::op()` is only
instantiated in the `else` branch, where `mpi_op_for` already guarantees `SBuf` is a range.

### Usage in `core::reduce` (and other reduction collectives)

```cpp
template <send_buffer SBuf, recv_buffer RBuf, typename Op>
    requires mpi_op_for<Op, SBuf>
void reduce(SBuf&& sbuf, RBuf&& rbuf, Op&& op, int root, MPI_Comm comm) {
    MPI_Reduce(
        ranges::data(sbuf), ranges::data(rbuf), ranges::size(sbuf),
        ranges::type(sbuf), native_op<SBuf>(op), root, comm
    );
}
```

No `if constexpr` at the collective level — op resolution is fully encapsulated in `native_op`.
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

- [x] **`core::barrier`** / **`v2::barrier`**
- [x] **`core::bcast`** / **`v2::bcast`**
- [x] **`core::allgather`** / **`v2::allgather`**
- [x] **`core::allgatherv`** / **`v2::allgatherv`**
- [x] **`core::alltoall`** / **`v2::alltoall`**
- [x] **`core::alltoallv`** / **`v2::alltoallv`**
- [ ] **Blocking**: `allreduce`, `scatter`, `scatterv`, `gather`, `gatherv`, `reduce`, `scan`, `exscan`
- [ ] **Non-blocking**: `iallreduce`, `iallgather`, `iallgatherv`, `ialltoall`, `ialltoallv`, `iscatter`, `iscatterv`, `igather`, `igatherv`, `ireduce`, `iscan`, `iexscan`, `ibarrier`
- Follow the same layering as p2p: `core::` wraps MPI directly, `v2::` handles inference and result types

### Collective interface design

#### Inplace single-buffer overloads (symmetric collectives only)

Single-buffer = inplace, but **only** for collectives where all ranks have the same
relationship to the buffer. For asymmetric collectives (gather, scatter, reduce) the buffer
plays different roles on root vs non-root (different semantics, different sizes) — no
single-buffer form there.

| Collective | Single-buffer form | Notes |
|---|---|---|
| `allreduce(buf, op, comm)` | ✓ | All ranks transform in place |
| `scan(buf, op, comm)` | ✓ | All ranks transform in place |
| `exscan(buf, op, comm)` | ✓ | All ranks transform in place |
| `allgather(buf, comm)` | ✓ | All ranks: local data already at `rank*n` slot, rest filled in |
| `gather` / `scatter` / `reduce` | ✗ | Buffer role differs by rank — always two-buffer |

The inplace allgather precondition (local data pre-placed at `rank * n`) is implicit and
documented but not enforced. For allreduce/scan/exscan, in-place is fully transparent.

`v2::inplace` sentinel is still available as an explicit send argument in two-buffer forms
for advanced use; most users never need it.

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
