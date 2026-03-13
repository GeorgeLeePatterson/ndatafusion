# Architecture

## Objective

Build `ndatafusion` into a production-grade DataFusion extension crate that exposes `nabled`
capabilities through explicit Arrow/DataFusion contracts while preserving `nabled`'s numerical
semantics and `ndarrow`'s Arrow interoperability rules.

This originally included prerequisite lower-layer hardening in `ndarrow` and `nabled` before
serious local `ndatafusion` implementation began. Those prerequisites are now satisfied by
published `ndarrow 0.0.3` and `nabled 0.0.7`.

## Definition Of "Complete And Ready For A V1 Publish"

`ndatafusion` is ready for a v1 publish when all of the following are true:

1. `Cargo.toml` forwards every `nabled` feature cleanly.
2. The crate exports a stable `register_all` entrypoint plus discoverable Rust helpers and UDF
   constructors.
3. The admitted v1 numerical domains are exposed through explicit DataFusion contracts with clear
   shape, null, and result semantics.
4. The implementation delegates to canonical `nabled` behavior rather than forking algorithms.
5. Required upstream releases of `ndarrow` and `nabled` exist with the contract and performance
   fixes that make `ndatafusion`'s SQL-facing assumptions valid.
6. Integration tests cover the admitted feature/value matrix and line coverage remains above 90%.
7. Docs, examples, and release metadata are complete enough for a first external consumer to
   succeed without reading the source.

## Core Architectural Insight

`ndarrow`, `nabled::arrow`, and DataFusion operate at different levels:

1. `ndarrow` owns the Arrow/ndarray boundary contract.
2. `nabled::arrow` is object-oriented or batch-oriented depending on the admitted lower-layer API:
   - one Arrow value represents one mathematical object
   - some newer APIs should instead operate over whole Arrow arrays that preserve row semantics
3. DataFusion UDFs are column-oriented:
   - one input array represents many row values
   - example: a `FixedSizeListArray` column can hold one vector per row

`ndatafusion` therefore should not assume that row-by-row lifting is the default architecture.
The correct execution order is:

1. use direct batch-native lower-layer Arrow contracts when they exist and are efficient
2. use a DataFusion cell-codec layer only for residual workflows that do not yet have a direct
   batch-native lower-layer contract
3. assemble DataFusion result columns only where the output contract naturally requires it

This makes the codec layer important, but no longer the default hot path.

## Pre-Architecture Gate: Upstream Hardening

The lower-layer pre-architecture gate is now satisfied:

1. `ndarrow`
   - released as `0.0.3` with the concept-first bridge contract closed for the required v1 matrix
2. `nabled::arrow`
   - released as `0.0.7` with the required batch-native Arrow façade extensions in place

`ndatafusion` should now wire dependencies, implement shared metadata, and prefer direct
batch-native delegation over fallback lifting.

## Canonical Concept Ingress Matrix

The primary cross-layer contract is concept-first:

1. rows are mathematical object families, not Arrow container categories
2. each family should have one canonical `rows-of-X` carrier
3. standalone single-object ingress may still exist, but batch execution should prefer the same
   `rows-of-X` carrier that `ndatafusion` will use

Current target matrix:

| Concept family | Standalone ingress | Canonical `rows-of-X` ingress | Current implication |
|---|---|---|---|
| Dense vector | one vector object | `FixedSizeList<T>(D)` | Stabilized and available for direct lower-layer delegation |
| Ragged vector / multivector | one ragged vector-like object | variable-shape tensor contract | Must stay distinct from dense-vector batching |
| Sparse vector | one sparse vector object | CSR rows | Naturally batch-shaped; higher-level API coverage still needs work |
| Dense matrix | one matrix object | fixed-shape tensor rank-2 carrier | Standalone matrix APIs may remain, but batch form should be canonical |
| Sparse matrix | one sparse matrix object | one sparse-matrix object per row | Stabilized via `ndarrow.csr_matrix_batch` and `nabled::arrow` batch wrappers |
| Fixed-shape tensor | one tensor object | `arrow.fixed_shape_tensor` | Structurally strong lower-layer fit |
| Variable-shape tensor | one ragged tensor object | `arrow.variable_shape_tensor` | Stabilized and available for direct lower-layer delegation |
| Complex vector | one complex vector object | first-class complex vector batch carrier | Stabilized through `ndarrow` and `nabled::arrow` |
| Complex matrix | one complex matrix object | first-class complex matrix batch carrier | Stabilized through `ndarrow` and `nabled::arrow` |
| Complex tensor | one complex tensor object | first-class complex tensor batch carrier | Stabilized through `ndarrow` and `nabled::arrow` |

## Fallback Lifted Value Contracts

The fallback v1 plan is to lift `nabled`'s existing object-level Arrow contracts into DataFusion
row values without inventing a competing shape system.

Whenever `ndarrow` and `nabled::arrow` admit a direct batch-native contract for a given domain,
that direct form is preferred instead of these lifted contracts.

### Dense Vector Values

1. `nabled` object contract: `PrimitiveArray<T>`
2. `ndatafusion` row contract: one row of `FixedSizeList<T>`
3. extraction rule: row slice -> `PrimitiveArray<T>` view-compatible object

### Dense Matrix Values

1. `nabled` object contract: `FixedSizeListArray`
2. `ndatafusion` row contract: one row of nested `FixedSizeList<FixedSizeList<T>>`
3. extraction rule: row slice -> `FixedSizeListArray` representing one matrix

### Dense Tensor Values

1. `nabled` object contract: canonical `arrow.fixed_shape_tensor`
2. `ndatafusion` row contract: one row carrying the same fixed-shape tensor metadata contract
3. extraction rule: row slice + field metadata -> one tensor object view

### Sparse CSR Values

1. `nabled` object contract: canonical `ndarrow.csr_matrix` extension or equivalent explicit
   sparse struct contract
2. `ndatafusion` row contract: one struct-valued row carrying the same sparse metadata
3. extraction rule: row slice + field metadata -> one sparse matrix object view

### Complex Dense Values

1. preserve `ndarrow`'s extension-aware complex encodings
2. do not collapse complex values into ad hoc `Struct<real, imag>` contracts unless that decision
   is explicitly revisited

## Delegation Model

Preferred execution should follow this flow:

1. validate DataFusion input fields, nested shapes, and extension metadata
2. if a direct batch-native `nabled::arrow` contract exists, delegate whole-array execution there
3. otherwise extract rows into object-level Arrow values compatible with `nabled` contracts
4. delegate to the canonical `nabled::arrow` function when one exists
5. if no `nabled::arrow` helper is appropriate, decode to ndarray views and call the canonical
   `nabled` ndarray-native public/view API directly
6. encode the result into builders for the final DataFusion output column only as required by the
   return contract

This keeps `ndatafusion` focused on DataFusion adaptation instead of algorithm ownership, and keeps
row-by-row lifting as a controlled fallback instead of the default design center.

## Public Extension Surface

The crate should mirror the shape used by standalone DataFusion function libraries:

1. `register_all(&mut dyn FunctionRegistry) -> Result<()>`
2. `functions` module for direct Rust call sites where that is useful
3. `udfs` module exporting individual `Arc<ScalarUDF>` constructors
4. optional `rewrite` and `planner` modules only when needed

V1 should be `ScalarUDF` first.

Use additional seams only for explicit value:

1. `FunctionRewrite` for constructor sugar or expression normalization
2. `ExprPlanner` only when SQL syntax cannot naturally reach the desired contract
3. `AggregateUDF`, `WindowUDF`, or table functions only after the scalar-UDF-first surface is
   stable and clearly insufficient

## Proposed Module Layout

Target crate layout:

```text
.
├── src/
│   ├── lib.rs
│   ├── register.rs
│   ├── error.rs
│   ├── metadata.rs
│   ├── signatures.rs
│   ├── codecs/
│   │   ├── vector.rs
│   │   ├── matrix.rs
│   │   ├── tensor.rs
│   │   ├── sparse.rs
│   │   └── complex.rs
│   ├── builders/
│   │   ├── primitive.rs
│   │   ├── list.rs
│   │   ├── tensor.rs
│   │   ├── sparse.rs
│   │   └── structs.rs
│   ├── udf/
│   │   ├── constructors.rs
│   │   ├── vector.rs
│   │   ├── matrix.rs
│   │   ├── decomposition.rs
│   │   ├── sparse.rs
│   │   ├── tensor.rs
│   │   └── ml.rs
│   ├── rewrite.rs
│   └── planner.rs
└── docs/
```

Notes:

1. no `mod.rs`
2. `metadata.rs` owns canonical `Field` and `DataType` builders
3. `codecs/*` owns residual row extraction from DataFusion columns into object-level Arrow values
   when direct batch delegation is unavailable
4. `builders/*` owns output assembly back into DataFusion columns where the return contract
   requires it
5. `udf/*` owns domain-specific `ScalarUDFImpl` implementations

## Result Contract Families

The v1 function catalog should use only a few predictable result shapes:

1. scalar outputs:
   - norms, determinant, rank, condition number, loss-like scalars
2. vector outputs:
   - solve results, singular values, projections, optimization iterates where natural
3. matrix outputs:
   - inverses, transforms, reconstructions, dense kernel outputs
4. tensor outputs:
   - fixed-shape tensor outputs where the shape is explicit and cheap to preserve
5. struct outputs:
   - decompositions and model-like workflows with multi-output results

Struct outputs are important for v1:

1. they keep related outputs together
2. they preserve semantics better than arbitrary sibling columns
3. DataFusion already supports struct-valued UDF results and field extraction

## Null And Error Policy

V1 should keep the contract strict and unsurprising:

1. null whole values propagate to null outputs by default
2. inner nulls inside numerical payloads are rejected
3. invalid shapes, incompatible nested list lengths, or broken extension metadata are execution
   errors
4. `nabled` errors should map into stable, domain-aware DataFusion execution errors
5. contract tightening in upstream layers must not silently degrade performance or correctness on
   previously admitted public APIs

## V1 Domain Scope

V1 should cover admitted capability parity across these domains:

1. constructors and normalizers
2. dense vector
3. dense matrix
4. decomposition and solver helpers
5. sparse
6. tensor
7. iterative, jacobian, optimization, PCA, regression, and stats

This is intentionally capability parity, not overload parity:

1. expose one good DataFusion-facing contract per workflow
2. do not mirror every `_view`, `_into`, or workspace helper from `nabled`

## Sequenced Plan To V1

### Phase 0: Governance Baseline

1. create docs and AGENTS tracking
2. lock architecture, constraints, and v1 scope

### Phase 1: `ndarrow` Hardening

1. make the concept-family `rows-of-X` matrix explicit in the Arrow/ndarray bridge contract
2. make nullable numerical payload contracts explicit
3. close complex and tensor shape gaps needed downstream
4. verify no correctness or performance regressions on already admitted public shapes

### Phase 2: `nabled::arrow` Hardening

1. add batch-native Arrow APIs that preserve DataFusion row/value semantics efficiently
2. make the canonical `rows-of-X` carrier explicit per concept family
3. add null-aware wrappers using the explicit `ndarrow` contracts
4. align public Arrow/view entrypoints with lower dispatched kernels where possible
5. preserve broader lower-layer public APIs that remain valid outside `ndatafusion`

## Release Checkpoints

Before local `ndatafusion` implementation resumes, two upstream checkpoints must be closed:

1. Checkpoint 1: `ndarrow` release-ready
   - concept-family `rows-of-X` carriers are explicit for the admitted bridge surface
   - remaining required bridge gaps are closed for the next `nabled` release
   - docs/tests/coverage are green for a new `ndarrow` release
2. Checkpoint 2: `nabled` release-ready
   - `nabled::arrow` exposes the required concept-first batch-native surface on top of checkpoint 1
   - docs/tests/coverage are green for a new `nabled` release
   - only then should `ndatafusion` wire those releases and proceed

### Phase 3: Dependency And Feature Contract

1. add `nabled`
2. add `ndarrow`
3. forward every `nabled` feature through `ndatafusion`
4. lock compile/test matrices for feature combinations

### Phase 4: Shared Contract Layer

1. implement `metadata.rs`
2. implement direct batch delegation paths for stabilized lower-layer Arrow contracts
3. implement output builders and common error mapping
4. implement row extraction codecs only for residual lifted vector, matrix, tensor, sparse, and
   complex workflows that still need them

### Phase 5: First Useful Slice

1. implement `register_all`
2. implement `functions` and `udfs`
3. land vector and matrix scalar UDFs first
4. add constructors or normalizers needed to use those functions from SQL

### Phase 6: Decomposition And Model Contracts

1. define struct-valued result schemas
2. land decomposition and solver wrappers
3. add field-extraction examples and tests

### Phase 7: Capability Expansion

1. sparse
2. tensor
3. ML/stat
4. remaining admitted parity gaps

### Phase 8: Publish Hardening

1. README examples
2. integration coverage across feature/value matrix
3. coverage proof greater than 90%
4. release metadata and publish readiness review

## Post-V1 Expansion

Not part of v1 unless later admitted:

1. UDAFs and window functions
2. table functions
3. SQL operator sugar and custom planners where scalar UDFs are already sufficient
4. deeper performance fusion to amortize any residual fallback lift/delegation overhead
