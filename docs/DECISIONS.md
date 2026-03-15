# Decisions

## Locked Decisions

1. `ndatafusion` is a DataFusion-facing facade over `nabled`; it does not own new numerical
   kernels.
2. `nabled` remains the source of truth for algorithm semantics, provider/backend/kernel routing,
   error behavior, and admitted Arrow object contracts.
3. `ndatafusion` targets capability parity, not 1:1 overload parity with `nabled::arrow`.
4. `ScalarUDF` is the default extension mechanism for v1.
5. Multi-output workflows return explicit struct-valued results when a scalar result is not
   natural; they are not flattened into ad hoc sibling columns.
6. Constructor and normalization functions are part of v1 scope because SQL users need an explicit
   path from ordinary nested values into the admitted numerical contracts.
7. UDAFs, window functions, table functions, planners, and rewrites are not required for v1 unless
   they unlock a natural SQL contract that scalar UDFs cannot express cleanly.
8. Whole-cell null inputs propagate to null outputs by default; inner nulls inside admitted
   numerical values are rejected as execution errors unless a function documents different
   behavior.
9. Preserve extension metadata on both input and output fields when the contract depends on it.
10. No hidden copy-heavy conversions in hot paths; explicit output materialization is acceptable
    when the function's Arrow/DataFusion result contract requires it.
11. `ndatafusion` must expose and forward all `nabled` feature flags cleanly.
12. Compatibility work must keep `datafusion`, Arrow, `ndarrow`, and `nabled` aligned on the same
    Arrow major version.
13. Upstream contract and performance hardening in `ndarrow` and `nabled` comes before meaningful
    local `ndatafusion` implementation.
14. Direct batch-native delegation into lower-layer Arrow APIs is preferred whenever the lower
    layers admit the relevant shapes efficiently; row/cell lifting is fallback-only.
15. Cross-repo hardening work must not introduce correctness or performance regressions in already
    admitted lower-layer behavior.
16. Existing broader `nabled::arrow` public APIs may remain when they serve valid non-`ndatafusion`
    use cases; `ndatafusion` should adapt to the canonical lower-layer surface instead of forcing
    that surface to collapse to DataFusion-only assumptions.
17. Numerical object nullability is row-null by default; inner nulls inside admitted numerical
    payloads are rejected unless a contract explicitly states otherwise.
18. `PrimitiveArray<T>` vector APIs in `nabled::arrow` are treated as object-level APIs, not as
    the default row-preserving `ndatafusion` scalar-UDF surface for primitive SQL columns.
19. Mathematical concept families are the primary ingress design unit: dense vector, ragged
    vector/multivector, sparse vector, dense matrix, sparse matrix, fixed-shape tensor,
    variable-shape tensor, and complex vector/matrix/tensor.
20. Each concept family should have one canonical `rows-of-X` batch carrier that serves both
    `ndatafusion` row semantics and standalone batch workflows in lower layers.
21. Ad hoc collections of standalone Arrow objects (for example `Vec<PrimitiveArray<T>>`) are not
    canonical batch carriers.
22. `ndatafusion` currently targets published `nabled 0.0.7` and `ndarrow 0.0.3`.
23. `ndatafusion` mirrors `nabled` feature names one-for-one rather than inventing a parallel
    feature taxonomy.
24. The admitted local real-valued catalog supports `Float32` and `Float64` over the implemented
    surface; `nabled/arrow` is enabled unconditionally because it is the base contract of
    `ndatafusion`.
25. When `nabled::arrow` already admits a direct batch-native hot path, `ndatafusion` should use
    it; when it does not, `ndatafusion` may iterate canonical batch carriers as ndarray views
    rather than invent a broad generic row codec prematurely.
26. Callback-driven `nabled::arrow` APIs such as numerical Jacobian/gradient/Hessian are not
    rejected as unimportant; they are deferred until `ndatafusion` has a SQL-natural way to
    specify the inner function being differentiated.
27. The first acceptable SQL contracts for callback-driven differentiation are:
    - specialized built-ins where the differentiated function is fixed and obvious
    - named-function registry contracts such as `jacobian('softmax', x)` after the underlying
      function family is admitted into `ndatafusion`
28. `ndatafusion` should not attempt to expose generic Rust-callback differentiation APIs directly
    as scalar UDFs; if a richer expression-driven differentiation model is added later, it should
    be an explicit planner/rewrite design, not an accidental scalar-UDF overload.
29. The first constructor surface is explicit `make_*` scalar UDFs over SQL `List` values plus
    scalar dimensions or shape lists; planner sugar and alternate constructor syntax are deferred
    until the base contracts stabilize.
30. `ndatafusion` remains git-consumed until DataFusion has a published Arrow-58-compatible
    crates.io release. README install guidance, crate docs, and publish checklists must not imply
    crates.io availability while the `datafusion` dependency is still pinned to git.
31. When SQL functions take data operands followed by control arguments, the preferred call style
    is positional data operands first and named trailing control arguments after them.
32. Custom coercion augments the explicit `make_*` constructors; it does not replace the
    structural constructor boundary for canonical Arrow contracts.
33. The first admitted aggregate surface is:
    - `vector_covariance_agg`
    - `vector_correlation_agg`
    - `vector_pca_fit`
    - `linear_regression_fit`
34. Grouped model fitting and grouped summary statistics should use `AggregateUDF` when they are
    naturally defined over many row observations rather than one row at a time.
35. Aggregate implementations must not treat raw-row accumulation plus opaque binary state as the
    default design pattern. If an exact mergeable sufficient-statistics state exists, it is the
    required implementation strategy.
36. Aggregate state should be typed, algebraic, and mergeable:
    - prefer explicit `state_fields()` over opaque `Binary` state payloads
    - keep dtype-specific execution behind a narrow runtime dispatch boundary
    - materialize Arrow outputs only at `evaluate`
    - isolate any fallback row-materialization strategy as an explicitly justified exception
37. The first aggregate redesign target is the current grouped fit/statistics wave:
    - `vector_covariance_agg`
    - `vector_correlation_agg`
    - `vector_pca_fit`
    - `linear_regression_fit`
    These should converge on typed sufficient-statistics state before more non-scalar surface area
    is added.

## Cross-Layer Contract Model

`nabled::arrow` contains both object-oriented and batch-oriented APIs.
`ndatafusion` is column-oriented: one DataFusion array holds many row values.

The preferred `ndatafusion` execution model is therefore:

1. use direct batch-native lower-layer contracts whenever they exist and preserve the natural
   DataFusion row/value semantics efficiently
2. use lifted cell contracts only when the lower layers do not yet admit the necessary batch-native
   form

## Concept-First Ingress Matrix

The canonical question is not "what Arrow type do we have?" but "what mathematical object family is
this?".

For each concept family:

1. a standalone single-object ingress may remain for ergonomics
2. one canonical `rows-of-X` carrier should exist for batch execution
3. `ndatafusion` should map to the canonical `rows-of-X` carrier whenever possible

Current target matrix:

1. Dense vector:
   - standalone: one vector object
   - canonical `rows-of-X`: `FixedSizeList<T>(D)` for fixed-dimension vectors
2. Ragged vector / multivector:
   - standalone: one ragged vector-like object
   - canonical `rows-of-X`: `arrow.variable_shape_tensor`-style ragged tensor carrier, not dense
     vector storage
3. Sparse vector:
   - standalone: one sparse vector object
   - canonical `rows-of-X`: CSR rows
4. Dense matrix:
   - standalone: one matrix object
   - canonical `rows-of-X`: fixed-shape tensor rank-2 carrier
5. Sparse matrix:
   - standalone: one sparse matrix object
   - canonical `rows-of-X`: one sparse-matrix object per row under an explicit batch contract
6. Fixed-shape tensor:
   - standalone: one fixed-shape tensor
   - canonical `rows-of-X`: `arrow.fixed_shape_tensor`
7. Variable-shape tensor:
   - standalone: one ragged tensor
   - canonical `rows-of-X`: `arrow.variable_shape_tensor`
8. Complex vector / matrix / tensor:
   - standalone: one complex object
   - canonical `rows-of-X`: first-class complex batch carriers

The fallback lifted cell contracts are derived from `nabled`'s object-level Arrow contracts:

1. Dense vector object:
   - `nabled` object contract: `PrimitiveArray<T>`
   - `ndatafusion` cell contract: one row of `FixedSizeList<T>`
2. Dense matrix object:
   - `nabled` object contract: `FixedSizeListArray`
   - `ndatafusion` cell contract: one row of nested `FixedSizeList<FixedSizeList<T>>`
3. Dense tensor object:
   - `nabled` object contract: canonical `arrow.fixed_shape_tensor`
   - `ndatafusion` cell contract: one row carrying the same explicit fixed-shape tensor contract
4. Sparse CSR object:
   - `nabled` object contract: canonical `ndarrow.csr_matrix` extension or equivalent explicit
     struct contract
   - `ndatafusion` cell contract: one row carrying the same sparse contract
5. Complex dense values:
   - use the same extension-aware fixed-size-list encoding that `ndarrow` and `nabled` already
     admit

This means `ndatafusion` may own row extraction and result assembly for residual cases, but should
delegate whole-array execution to the canonical `nabled` Arrow or ndarray path whenever possible.

## API Purity Model

1. Public Rust helpers in `ndatafusion` should expose DataFusion/Arrow-native contracts.
2. Public numerical behavior should still be inherited from `nabled`, not re-described in a new
   algorithm layer.
3. SQL-facing function names should be concise and domain-oriented, not provider-specific.
4. `ndatafusion` may expose a narrower, more SQL-natural contract surface than `nabled::arrow`
   without requiring `nabled::arrow` to remove broader existing APIs.
5. Standalone batch workflows in lower layers should prefer the same canonical `rows-of-X`
   carriers that `ndatafusion` will use.

## Near-Term Non-Goals

1. Re-implementing `nabled` algorithms directly inside `ndatafusion`.
2. Runtime provider/backend dispatch APIs that do not exist in `nabled`.
3. Ad hoc SQL sugar before the core value contracts, registration story, and dense-function slices
   are stable.
4. Publishing with dependency drift or feature mismatch relative to `nabled`.
