# Status Snapshot

Last updated: 2026-03-15

## Summary

`ndatafusion` is now well past the initial v1 target.

Current validated surface:

1. `147` scalar UDFs registered through `register_all`
2. `4` aggregate UDFs registered through `register_all`
3. `1` table function, `unpack_struct`, registered through `register_all_session`
4. real-valued support across canonical `Float32` and `Float64` dense vector, dense matrix,
   sparse CSR, fixed-shape tensor, variable-shape tensor, matrix-equation, statistics, grouped
   aggregate, and tensor-decomposition workflows
5. complex-valued support across canonical `ndarrow.complex64` dense vector, dense matrix,
   complex PCA, fixed-shape tensor, and variable-shape tensor workflows
6. named arguments, aliases, custom scalar coercion, programmatic `documentation()`, per-UDF
   simplify hooks, retractable aggregates, and a table-function entry point for struct unpacking

Validation on the current tree:

1. `just checks` passes
2. line coverage is `90.01%`
3. `cargo doc --no-default-features --no-deps` passes

## Current Surface

Implemented and covered today:

1. constructor UDFs for canonical vector, matrix, fixed-shape tensor, variable-shape tensor, and
   CSR sparse-batch contracts
2. dense vector, dense matrix, decomposition, direct-solver, spectral, matrix-function, and
   matrix-equation UDFs across the admitted real-valued surface
3. complex vector, matrix, complex PCA, complex tensor, and complex spectral/matrix-function UDFs
4. sparse factorization and preconditioner UDFs using explicit struct-valued state contracts
5. named-function differentiation UDFs: `jacobian`, `jacobian_central`, `gradient`, `hessian`
6. named-function complex optimization UDFs
7. tensor decomposition and tensor-train UDFs
8. grouped aggregate UDFs for covariance, correlation, PCA fit, and linear regression fit
9. ordered window use of the current aggregate wave through retractable aggregate state
10. `unpack_struct` as a generic one-row table-function bridge for struct-valued scalar results

## Release Posture

`ndatafusion` is ready for continued git consumption.

Crates.io publication is still blocked by one external issue:

1. `datafusion` is still pinned to an Arrow-58-compatible git revision because the published
   crates.io line does not yet satisfy the Arrow 58 contract required by `ndarrow` and `nabled`

## Next Directions

The remaining work is no longer foundational implementation. It is planning and selective
expansion work:

1. broader planner hooks beyond `simplify`, only where they materially help optimization
2. custom expression planning for SQL forms that do not fit ordinary scalar, aggregate, or table
   functions
3. richer table-function surfaces only where a relation-shaped contract is genuinely better than a
   struct-valued scalar result
4. optional dedicated `WindowUDF` surfaces only if retractable aggregates become insufficient
5. crates.io publication once DataFusion exposes a compatible published release line
