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

## Clean layer split (namespace + file hierarchy)

The four-layer architecture described in CLAUDE.md is not yet reflected in the file layout or namespaces. Currently both bridge-layer and language-bindings-layer code live under `include/kamping/v2/ranges/` and `include/kamping/v2/views/` without a clear boundary.

### Goal

Enforce the split so that headers in the bridge layer never include language-bindings headers, and the layer each file belongs to is obvious from its path.

Proposed layout sketch:
```
include/kamping/v2/
  bridge/
    concepts.hpp        ← ranges/concepts.hpp (buffer concepts, no deferred)
    ranges.hpp          ← ranges/ranges.hpp (accessor dispatch)
    adaptor.hpp         ← ranges/adaptor.hpp (pipe machinery)
    view_interface.hpp  ← core half of view_interface (see below)
    all.hpp             ← ref_view / owning_view / all() (see below)
    views/              ← with_type, with_size, with_counts, with_displs (core views)
  ...                   ← language-bindings layer (infer, resize_view, auto_counts, …)
```

### `view_interface` split

`ranges/view_interface.hpp` currently forwards **all** protocol methods — both the core
buffer metadata (`mpi_type`, `mpi_size`, `mpi_data`, `mpi_sizev`, `mpi_displs`, `counts`, `displs`)
and the deferred-buffer protocol (`mpi_resize_for_receive`, `commit_counts`, `set_comm_size`,
`displs_monotonic`). The deferred protocol belongs to the language-bindings layer.

Proposed split:
- **`view_interface_core`** (bridge layer) — forwards only metadata accessors; no deferred methods.
  Core views (`with_counts_view`, `with_displs_view`, `with_type_view`, `with_size_view`) inherit
  from this.
- **`view_interface`** (language-bindings layer, extends `view_interface_core`) — adds the deferred
  forwarding methods. Language-binding views (`resize_view`, `auto_counts_view`,
  `auto_displs_view`, `resize_v_view`, `flatten_v_view`) inherit from this.

### `ref_view` / `owning_view` placement

Both layers need `ref_view`/`owning_view`:
- Bridge-layer core views use them (via `all()`) to wrap the base range when building a view chain.
- Language-bindings views also use them — and crucially, when the wrapped type `T` has the
  deferred protocol, the wrapping `ref_view`/`owning_view` must forward it (so that, e.g.,
  an `owning_view<resize_view<…>>` stored in `iresult` still exposes `mpi_resize_for_receive`).

#### Why option (a) / two tiers doesn't work

The deferred protocol must propagate through the *entire* view chain to reach the underlying
container. Consider `vec | with_type(t) | resize`:

```
resize_view
  └─ with_type_view<ref_view<vec>>   ← resize_view calls resize_for_receive(base_, n) here
       └─ ref_view<vec>
            └─ vec
```

`resize_view::mpi_data()` calls `kamping::ranges::resize_for_receive(base_, n)` which dispatches
to `base_.mpi_resize_for_receive(n)` or `base_.resize(n)`. If `with_type_view` uses
`view_interface_core` (no deferred forwarding), it exposes neither — the chain is broken and
the code fails to compile. The same failure applies to `commit_counts`/`set_comm_size` propagating
through `with_counts_view` or `with_displs_view`. Any core view in the middle of a mixed chain
becomes an opaque barrier.

Option (a) is only viable if core views are always outermost (applied *after* all deferred views),
which is the reverse of natural composition order and cannot be enforced at compile time.

#### Preferred approach: template view-interface parameter

Make each core view accept a template-template parameter for its view interface base, defaulting
to `core::view_interface`. The v2 layer then instantiates the same view implementation with
`v2::view_interface`, which adds the deferred forwarding. No logic is duplicated; `v2::with_type_view`
is a type alias:

```cpp
// core layer
template <typename Base, template <typename> class VI = core::view_interface>
class with_type_view : public VI<with_type_view<Base, VI>> { ... };

// v2 layer — alias, no new logic
template <typename Base>
using with_type_view = core::with_type_view<Base, v2::view_interface>;
```

The same parameter must be applied to `ref_view`, `owning_view`, and `all()`, since views
wrap their base via `all()` internally:

```cpp
template <typename T, template <typename> class VI = core::view_interface>
class ref_view : public VI<ref_view<T, VI>> { ... };

// all() parameterized — v2 views call v2::all() internally
template <template <typename> class VI = core::view_interface, typename R>
constexpr auto all(R&& r) { ... }
```

With this design, `vec | v2::with_type(t) | resize` produces:

```
resize_view< v2::with_type_view< v2::ref_view<vec> > >
```

`v2::with_type_view` has `mpi_resize_for_receive` (via `v2::view_interface`) → forwarded from
`v2::ref_view` → forwarded from `vec.resize()` ✓.

Meanwhile `core::with_type_view` (used by `core::send`/`core::recv`) uses `core::view_interface`
and compiles without any v2 headers, making core truly standalone.

**Important footgun**: every view that wraps a base via `all()` internally must call the
appropriately parameterized `all()`. A v2 view that accidentally calls `core::all()` will wrap
in a `core::ref_view` and silently block deferred protocol propagation. This needs to be
enforced by convention or by having v2 views unconditionally call `v2::all()`.

**Prerequisite**: the deferred-protocol concepts (`has_commit_counts`, `has_set_comm_size`,
`has_mpi_resize_for_receive`, `has_monotonic_displs`) currently live in `ranges/concepts.hpp`
alongside the core buffer concepts. They must be moved to a separate header (e.g.
`bridge/deferred_concepts.hpp`) so that `core::view_interface` does not need to include them
and the core layer remains self-contained.

#### Alternative: single view_interface, include-discipline only (option 3)

Keep the current single `view_interface` with all forwarding. No include-level layering
violation today because the deferred concepts are already in `ranges/concepts.hpp` (bridge
layer) — `view_interface.hpp` has no upward dependency. The layer split is enforced purely
by convention: core views never include v2-only headers.

This is simpler and avoids the template-template parameter complexity and the `all()` footgun.
The cost: if core is ever extracted as a standalone library, `core::view_interface` will carry
forwarding methods for protocol methods it has no business knowing about.

**Decision**: use the template view-interface parameter approach if standalone core is a
real goal; use option 3 if the split remains a soft architectural principle within one repo.

#### Alternative: views-free core

A more radical alternative that sidesteps the entire view_interface split problem: **remove
all views from the core layer entirely**. Core should be a language-specific implementation
of the contract, not a C++ ergonomics layer. Pipe syntax and CRTP view machinery are ergonomic
choices that belong in v2.

**What stays in core:**
- `ranges/concepts.hpp` — buffer concepts (`data_buffer`, `send_buffer`, `recv_buffer`, `data_buffer_v`)
- `ranges/ranges.hpp` — accessor dispatch (`size/data/type/sizev/displs`), `buffer_traits`,
  contiguous-range → `MPI_Datatype` deduction
- `native_handle.hpp` — MPI handle extraction
- `core::send`, `core::recv`, etc. — bare MPI wrappers, one call each
- Two plain helper structs for cases where the caller needs to attach metadata that their
  type cannot express via `buffer_traits` (see below)

**What moves entirely to v2:**
- `view_interface`, `ref_view`, `owning_view`, `all()` — no longer needed by core
- Pipe adaptor machinery (`adaptor`, `adaptor_closure`, `composed_closure`)
- All `with_*` views (`with_type_view`, `with_counts_view`, `with_displs_view`, `with_size_view`)
- Everything already in v2 (`resize_view`, `auto_counts_view`, etc.)

**Core helper structs** — fully concrete, non-template, no CRTP, no pipe:

```cpp
// Non-owning buffer with explicit MPI type.
// void* covers both send (implicitly converts to void const*) and recv.
// Mirrors std::span extended with an MPI type; name follows MPI's own type-erased style.
struct mpi_span {
    void*          data;
    std::ptrdiff_t size;      // element count
    MPI_Datatype   type;
};

// Variadic buffer: data + per-rank counts and displacements.
// Name follows MPI's 'v'-suffix convention (allgatherv, alltoallv, …).
// MPI mandates int for counts/displs, so no template parameter needed.
struct mpi_span_v {
    void*          data;
    MPI_Datatype   type;
    int const*     counts;    // per-rank element counts
    int const*     displs;    // per-rank displacements
    int            comm_size;
};
```

Both satisfy their respective buffer concepts via member functions; no view machinery needed.

**The `with_size` / prefix-send use case:**
Use `std::span` / `std::views::take` to trim the range before passing to core, or construct
an `mpi_span` directly. There is no `views::with_size` in core.

**Benefits:**
- Core is genuinely standalone: no view headers, no CRTP, no pipe machinery. Usable by
  anyone comfortable with C++ ranges and MPI without learning the view adaptor system.
- The template-VI parameter problem and the `all()` footgun disappear entirely.
- `buffer_traits` remains the extension point for custom types in core.
- v2 views are free to use whatever view_interface design they want without any core coupling.
- **Reference implementation potential**: a views-free core, with simple concrete helper types
  and a clean buffer concept hierarchy, is legible to MPI implementors and WG members who are
  not C++ template experts. The paper aims to propose core as a candidate reference C++ MPI
  interface; that case is strongest when core does not depend on any particular ergonomics
  layer. Design decisions in core should be held to this standard: would an MPI WG find this
  adoptable as a language-bridge specification?

**Trade-offs:**
- `core::send(vec | with_type(MPI_BYTE), ...)` becomes
  `core::send(mpi_span{vec.data(), vec.size(), MPI_BYTE}, ...)`, or simply
  `core::send(vec, ...)` when the MPI type is deducible. Less composable at the core level,
  but that is intentional: composition is v2's job.
- `mpi_span_v` requires the caller to already hold counts and displs as flat `int` arrays,
  which is typical at the core level.

### Namespace alignment

Currently everything is under `kamping::ranges::` and `kamping::views::` regardless of layer.
Consider:
- `kamping::bridge::ranges::` / `kamping::bridge::views::` for the bridge layer
- `kamping::v2::views::` for the language-bindings layer (already partly true for `v2::` wrappers)
Or keep the flat namespaces and rely on file paths alone for layer documentation.

#### Re-evaluate `kamping::ranges::` as a namespace

`kamping::ranges::` was chosen to mirror `std::ranges::`, but a data buffer is not a range —
it is a distinct abstraction (flat memory region + MPI type, not an iterator pair). Using the
`ranges` namespace sub-communicates the wrong mental model to readers and to MPI WG reviewers.

If namespaces are made to reflect layers rather than mirror the std library, natural candidates
are:
- `kamping::core::` — for the buffer concepts, accessor dispatch, and helper structs (aligns
  with the existing `kamping::core::send` etc.)
- `kamping::v2::` — for the ergonomics layer (already used for wrappers)
- Drop `kamping::ranges::` and `kamping::views::` entirely once the layer split is done

The accessor free functions (`size`, `data`, `type`, …) and `buffer_traits` would then live
in `kamping::core::`, making it clear they are part of the core contract rather than range
adaptors. This also avoids the oddity of `kamping::ranges::size` shadowing `std::ranges::size`
in translation units that use both.

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

## Collectives

- [x] **`core::barrier`** / **`v2::barrier`**
- [x] **`core::bcast`** / **`v2::bcast`**
- [ ] **Blocking**: `allreduce`, `allgather`, `allgatherv`, `alltoall`, `alltoallv`, `scatter`, `scatterv`, `gather`, `gatherv`, `reduce`, `scan`, `exscan`
- [ ] **Non-blocking**: `iallreduce`, `iallgather`, `iallgatherv`, `ialltoall`, `ialltoallv`, `iscatter`, `iscatterv`, `igather`, `igatherv`, `ireduce`, `iscan`, `iexscan`, `ibarrier`
- Follow the same layering as p2p: `core::` wraps MPI directly, `v2::` handles inference and result types
