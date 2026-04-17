# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KaMPIng (Karlsruhe MPI next generation) is a **header-only C++17 MPI wrapper library** providing modern, near-zero-overhead bindings for MPI. It uses template metaprogramming with named parameters (Python-style) so callers can pass arguments in any order and omit parameters to get sensible defaults.

## Build Commands

Requires CMake 3.25+ and an MPI installation. Out-of-source builds are enforced.

```bash
# Recommended: use presets (CMake 3.20+)
cmake --preset release
cmake --build --preset release --parallel

cmake --preset debug
cmake --build --preset debug --parallel

# Traditional
cmake -B build -DCMAKE_BUILD_TYPE=Release -DKAMPING_BUILD_EXAMPLES_AND_TESTS=ON
cmake --build build --parallel
```

Key CMake options:
- `KAMPING_BUILD_EXAMPLES_AND_TESTS` — must be ON to build tests (default OFF in library mode)
- `KAMPING_ASSERTION_LEVEL` — `none`/`light`/`normal`/`heavy`/`light_communication`/`heavy_communication` (default: `normal` in Debug, `exceptions` in Release)
- `KAMPING_EXCEPTION_MODE` — use exceptions for MPI errors (default ON)
- `KAMPING_ENABLE_SERIALIZATION` — Cereal integration (default ON)
- `KAMPING_ENABLE_REFLECTION` — Boost.PFR struct reflection (default ON)

## Testing

```bash
# Run all tests
ctest --preset release          # or --preset debug
make -C build check             # alternative

# Run a single test by name
ctest --test-dir build -R test_<testname>
# Or build and run directly
cmake --build build --target test_<testname>
./build/tests/test_<testname>
```

MPI tests run with multiple core counts (1, 2, 4, 8). Test timeout defaults to 20s (`KAMPING_TEST_TIMEOUT`).

## Code Formatting

```bash
cmake --build build --target check-clang-format   # validate C++ formatting
cmake --build build --target check-cmake-format   # validate CMake formatting
```

- C++: `clang-format` with `.clang-format` — 120 column limit, 4-space indent
- CMake: `cmake-format` with `.cmake-format.py` — 120 column limit

Both are enforced in CI. Run formatters before committing.

## Architecture

### Core Design Pattern: Named Parameters

The central abstraction is a *named parameter* system. Instead of positional MPI arguments, callers pass typed parameter objects (`send_buf(...)`, `recv_buf(...)`, `send_counts(...)`, etc.) in any order. Template metaprogramming fills in omitted parameters at compile time with zero runtime overhead.

### Key Components

**`Communicator<DefaultContainerType, Plugins...>`** (`include/kamping/communicator.hpp`)
The main user-facing class wrapping `MPI_Comm`. Collective and p2p operations are implemented as member functions. Extensible via CRTP plugins (see `include/kamping/plugin/`).

**Named Parameters** (`include/kamping/named_parameters.hpp` + `named_parameter_*.hpp`)
Parameter factory functions that return typed objects encoding parameter identity. Collective implementations inspect the parameter pack at compile time to determine which parameters were provided and compute defaults.

**Data Buffers** (`include/kamping/data_buffer.hpp`)
Abstraction over send/receive buffers. Supports resize policies: `resize_to_fit`, `no_resize`, `grow_only`. Automatic mapping from C++ types to `MPI_Datatype`.

**Collective Operations** (`include/kamping/collectives/`)
One file per collective: `allgather.hpp`, `allreduce.hpp`, `alltoall.hpp`, `bcast.hpp`, `scatter.hpp`, `gather.hpp`, `reduce.hpp`, `scan.hpp`, `exscan.hpp`, etc. Non-blocking variants (`iallreduce`, `ibarrier`) included.

**Point-to-Point** (`include/kamping/p2p/`)
`send.hpp`, `recv.hpp`, `isend.hpp`, `irecv.hpp`, `sendrecv.hpp`, `probe.hpp`, `try_recv.hpp`.

**Plugin System** (`include/kamping/plugin/`)
CRTP mixins that extend `Communicator` with higher-level algorithms:
- `alltoall_sparse.hpp` / `alltoall_grid.hpp` / `alltoall_dispatch.hpp` — optimized alltoall strategies
- `reproducible_reduce.hpp` — bit-identical reductions
- `sort.hpp` — distributed sorting
- `ulfm.hpp` — User-Level Failure-Mitigation

**Type System** (`include/kamping/builtin_types.hpp`, `mpi_datatype.hpp`)
Automatic mapping from C++ arithmetic types and user-defined structs to `MPI_Datatype`.

**Error Handling** (`include/kamping/error_handling.hpp`)
`THROW_IF_MPI_ERROR` macro and configurable assertion levels via `kassert`.

**Requests** (`include/kamping/request.hpp`, `request_pool.hpp`)
RAII wrappers for non-blocking MPI requests.

**Measurements** (`include/kamping/measurement/`)
Hierarchical timer infrastructure with MPI-aware aggregation.

### Namespaces

- `kamping::` — public API
- `kamping::internal::` — implementation details (not stable API)

---

## KaMPIng v2 (`include/kamping/v2/`)

v2 is a ground-up redesign of the public API. The named-parameter / template-metaprogramming approach of v1 is replaced by a concept-based buffer protocol with composable range adaptors. v2 currently lives alongside the existing code; only `include/kamping/v2/` and `examples/v2/` are part of the new design.

### Layered Architecture

v2 is organized into four explicit layers. Each layer depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────┐
│  Ecosystem bridges                                       │
│  Bindings to external libraries (Cereal, …)             │
│  kamping/v2/contrib/                                     │
├─────────────────────────────────────────────────────────┤
│  Language bindings  (kamping-v2)                         │
│  C++ ergonomics: ownership, infer, deferred buffers,     │
│  auto-counts/displs, resize-on-receive                   │
│  kamping/v2/  (excluding contrib/)                       │
├─────────────────────────────────────────────────────────┤
│  Language bridge                                         │
│  Minimal C++ implementation of the buffer contract:      │
│  count/ptr/type/counts/displs dispatch,                  │
│  core view adaptors (with_counts, with_displs, …),       │
│  mpi::experimental:: MPI wrappers, native_handle bridge  │
│  include/mpi/, kamping/v2/ranges/, kamping/v2/views/     │
├─────────────────────────────────────────────────────────┤
│  Contract  (language-agnostic)                           │
│  Abstract description of send/recv buffer,               │
│  variadic buffer (with counts+displs), native MPI object │
│  — expressed as C++ concepts in ranges/concepts.hpp      │
└─────────────────────────────────────────────────────────┘
```

**Contract layer** — language-agnostic definitions of what a send buffer, recv buffer, variadic buffer (with counts and displacements), and a native MPI handle *are*. Expressed in C++ as the concepts in `kamping/v2/ranges/concepts.hpp` (`send_buffer`, `recv_buffer`, `data_buffer_v`, …) and as the `MPI_Comm`/`MPI_Request`/… handle family. No implementation here — only constraints.

**Language bridge** — the minimal C++ wiring that makes the contract work for the language:
- Accessor dispatch functions (`mpi::experimental::count/ptr/type/counts/displs`) with three-tier priority: `buffer_traits<T>` specialization → member functions (`mpi_count()`, `mpi_ptr()`, etc.) → `std::ranges` / builtin-type fallbacks.
- `mpi::experimental::buffer_traits<T>` and `native_handle_traits<T>` — non-intrusive customization points for third-party types.
- Ownership infrastructure: `ref_view<T>` (non-owning, wraps lvalue), `owning_view<T>` (owning, wraps rvalue), and `all(r)` / `all_t<R>` which select between them. This is foundational plumbing used by the adaptor machinery and by the bindings layer.
- View adaptor machinery: `view_interface_base`, `adaptor_closure`, `adaptor`, `composed_closure` — the pipe `|` operator infrastructure.
- Core view adaptors that carry metadata through a pipe without adding ownership or resize logic: `with_type`, `with_size`, `with_counts`, `with_displs`.
- `mpi::experimental::` MPI wrappers — one MPI call each, no inference, no resizing, throw `mpi_error` on failure. Concrete buffer types `mpi_span` and `mpi_span_v` for calling without the view pipeline.
- `mpi::experimental::native_handle` / `to_rank` / `to_tag` — extract raw MPI handles from any wrapper.

**Language bindings (kamping-v2)** — C++ ergonomics and MPI convenience on top of the bridge:
- `infer()` protocol: operation-tagged ADL hook that resolves unknown recv sizes (via `MPI_Mprobe`) or variadic counts before the MPI call is issued.
- Deferred buffer concepts (`deferred_recv_buf`, `deferred_recv_buf_v`) and the views that implement them: `resize`, `resize_v`, `auto_counts`. These exploit the bridge's `ref_view`/`owning_view` to hold buffers safely.
- `auto_displs` — computes displacements via `exclusive_scan`; tags result as `has_monotonic_displs` to enable O(1) resize.
- `iresult<Buf>` — move-only non-blocking handle; stores the buffer on the heap via `unique_ptr<all_t<Buf>>` so the pointer captured by MPI remains stable after a move.
- Sentinel buffers (`inplace`, `null_buf`, `bottom`) — zero-overhead special buffer values for collective shortcuts.
- `kamping::v2::` wrappers (send, recv, bcast, …) that call `infer()` then delegate to `mpi::experimental::`.

**Ecosystem bridges** — bindings to external C++ libraries, living in `kamping/v2/contrib/`. Currently: Cereal serialization via `views::serialize` / `views::deserialize<T>()`.

### Core Idea: Buffer Protocol + View Pipeline

Instead of wrapping arguments in named-parameter factory functions, callers pipe standard C++ objects through a chain of `std::ranges`-style view adaptors that attach MPI metadata (element count, MPI datatype, per-rank counts, displacements). The resulting view satisfies one of the buffer concepts and is passed directly to a free-function MPI wrapper.

```cpp
// v1 style
comm.send(send_buf(v), destination(1));

// v2 style — metadata attached via pipe
kamping::v2::send(v, 1, comm);
kamping::v2::recv(v | kamping::views::resize, comm);           // resizable recv
kamping::v2::send(map | kamping::views::serialize, 1, comm);   // Cereal serialization
```

### Namespaces

| Namespace | Role |
|-----------|------|
| `mpi::experimental::` | Core layer: buffer concepts, accessors (`count`, `ptr`, `type`, `counts`, `displs`), MPI wrappers, native-handle adaption, concrete buffer types (`mpi_span`, `mpi_span_v`, `comm_view`) |
| `kamping::v2::` | High-level wrappers: call `infer()` then delegate to `mpi::experimental::` |
| `kamping::views::` | Range adaptor factory functions (pipe operators) |

### Buffer Concepts (`include/mpi/buffer.hpp`)

A type satisfies a buffer concept when `mpi::experimental::count(t)`, `mpi::experimental::ptr(t)`, and `mpi::experimental::type(t)` return the right kinds of values:

| Concept | Requirements |
|---------|-------------|
| `data_buffer` | count + ptr (any pointer) + type |
| `send_buffer` | data_buffer with `ptr()` convertible to `void const*` |
| `recv_buffer` | data_buffer with `ptr()` convertible to `void*` |
| `data_buffer_v` | data_buffer + `counts()` (counts range) + `displs()` (displs range) |
| `deferred_recv_buf` | recv_buffer with `set_recv_count(n)` for late-bound sizes |
| `deferred_recv_buf_v` | variadic version with `set_comm_size`, `mpi_counts()`, `commit_counts()` |

### Accessor Dispatch (`include/mpi/buffer.hpp`)

Each of `count()`, `ptr()`, `type()`, `counts()`, `displs()` is a free function in `mpi::experimental::` with prioritized overload resolution:

1. `buffer_traits<T>` specialization — non-intrusive, for types you don't own
2. `t.mpi_count()` / `t.mpi_ptr()` / `t.mpi_type()` / `t.mpi_counts()` / `t.mpi_displs()` member functions
3. `std::ranges::size` / `std::ranges::data` + builtin MPI type deduction (for non-variadic buffers)

Specialize `mpi::experimental::buffer_traits<T>` to adapt any third-party type without modifying it.

### View Adaptors (`include/kamping/v2/views/`)

All views are composable with `|`. They are lazy; metadata is not computed until the MPI operation queries it.

| View factory | Effect |
|---|---|
| `views::resize` | Marks a container as resizable; MPI recv will call `resize(n)` before writing |
| `views::with_type(dt)` | Overrides the MPI datatype |
| `views::with_size(n)` | Overrides element count |
| `views::with_counts(range)` | Attaches per-rank send/recv counts (variadic operations) |
| `views::with_displs(range)` | Attaches per-rank displacements; pass `kamping::v2::monotonic` tag to enable O(1) resize |
| `views::auto_displs([tag,] [container])` | Computes displacements via exclusive_scan of counts; always `has_monotonic_displs` |
| `views::resize_v` | Variadic recv buffer: resizes the underlying container from counts+displs before receive |
| `views::ref_single(val)` | Wraps a single scalar as a one-element contiguous buffer |
| `views::auto_counts([buf])` | Deferred variadic counts buffer; `set_comm_size` / `commit_counts` protocol |
| `views::serialize` / `views::deserialize<T>()` | Cereal serialization (contrib) |

Ownership semantics follow `std::ranges::all`: an lvalue produces a `ref_view` (borrows); an rvalue produces an `owning_view` (owns). This propagates through the view chain.

### `infer()` Protocol (`include/kamping/v2/infer.hpp`)

Before issuing an MPI call, `kamping::v2::` wrappers call `infer(comm_op::XXX{}, buf..., comm)`. The default overloads:
- For `deferred_recv_buf` targets: probe the network (`MPI_Mprobe`) and call `set_recv_count(n)` so the buffer resizes before the actual receive.
- For variadic operations: call `set_comm_size` and let MPI write directly into the counts buffer.

Users can add new ADL overloads of `infer()` for custom buffer types or to exchange additional metadata.

### Non-Blocking Operations (`include/kamping/v2/iresult.hpp`)

Non-blocking operations return `iresult<Buf>` (single buffer) or `iresult<SBuf, RBuf>` (sendrecv).

- Move-only; the buffer is stored on the heap via `unique_ptr` so the pointer captured by MPI remains stable after a move.
- `.wait([status])` — blocks and returns the buffer (owned buffers moved out; borrowed buffers return an lvalue reference).
- `.test([status])` — polls; borrowed buffers return `bool`, owned buffers return `std::optional<T>`.
- Destructor calls `MPI_Wait` if the request was not already completed, preventing silent data corruption.

### Native Handle Bridge (`include/kamping/v2/native_handle.hpp`)

Free functions `mpi::experimental::native_handle(x)` and `native_handle_ptr(x)` extract `MPI_Comm`, `MPI_Request`, etc. from arbitrary wrapper types. Dispatch priority:

1. `native_handle_traits<T>` specialization
2. `t.mpi_native_handle()` / `t.mpi_native_handle_ptr()` member functions
3. Passthrough for raw `MPI_Comm` / `MPI_Request` / … values

The same pattern applies to ranks (`to_rank`) and tags (`to_tag`), supporting both plain `int` and strongly-typed wrappers.

### Coding Conventions

- Header-only: all implementation in `.hpp` files under `include/kamping/`
- `#pragma once` for include guards
- East-side `const` (`int const` not `const int`)
- CamelCase for types, snake_case for functions/variables
- Private members prefixed with `_`: `_member`
- Doxygen required for all public API
- No `using namespace` in headers
- `auto` preferred for type deduction
