# Status Snapshot

Last updated: 2026-03-13

## Summary

`ndatafusion` is now past the planning-only baseline, and local implementation work has started on
top of the released upstream contracts.

1. The repository is currently a single crate with a thin but real module surface.
2. DataFusion is pinned to an Arrow-58-compatible git revision on `main`.
3. Local quality gates pass on the current scaffold.
4. Planning docs and AGENTS tracking instructions now exist.
5. Published `nabled 0.0.7` and `ndarrow 0.0.3` are now wired in.
6. `ndatafusion` mirrors `nabled`'s feature flags one-for-one.
7. A real local registration/catalog surface now exists via `register_all`, `functions`, `udfs`,
   shared metadata/error helpers, and domain modules.
8. A substantial `f64`-first numerical DataFusion UDF catalog is now implemented directly as the
   crate surface.
9. The current catalog now has constructor/helper tests, contract-edge validation, successful-path
   domain coverage, and line coverage above 90%.
10. The active ingress model remains concept-first: each mathematical object family needs one
   canonical standalone ingress and one canonical `rows-of-X` batch carrier.

## Current Repository Reality

1. Root files exist for crate metadata, linting, formatting, and a minimal integration test.
2. There is no `docs/` implementation history before this planning baseline.
3. A larger internal module tree now exists for registration, shared metadata/signature/error
   helpers, and domain UDFs.
4. Direct batch-native delegation now exists across the admitted vector, matrix, sparse, and
   tensor workflows whenever `nabled::arrow` exposes the needed batch carrier directly.
5. The fallback generic cell-codec layer still does not exist; residual unsupported workflows are
   currently handled case-by-case or remain unimplemented.
6. Current validation covers both successful batch-native paths and representative type, shape,
   scalar-argument, and batch-length failure contracts.

## Constraints In Force

1. `ndatafusion` is a facade over `nabled`, not a numerical fork.
2. Arrow/DataFusion value contracts must remain explicit and aligned with `ndarrow`.
3. Capability parity beats overload parity.
4. Quality gates remain strict: `just checks`, pedantic clippy compliance, and coverage greater
   than 90% before push.
5. `nabled` feature forwarding is a hard requirement for the published crate surface.
6. Cross-repo prerequisite work must preserve correctness and performance in already admitted
   lower-layer behavior.
7. Existing broader lower-layer public APIs do not need to be removed just because `ndatafusion`
   will depend on a narrower SQL-facing contract surface.

## Current Code Ownership

Today:

1. `src/lib.rs`, `src/register.rs`, `src/functions.rs`, `src/udfs.rs`
   - thin local registration and catalog scaffold only
2. `tests/e2e.rs`
   - placeholder integration target required by `.justfile`

Target ownership after the first real implementation rounds:

1. registration and public crate surface
2. shared metadata and cell codecs
3. domain UDF modules grouped by numerical domain
4. optional planner or rewrite integration only if admitted later

## Next Required Milestone

Local implementation round:

1. Add SQL constructors and normalizers for canonical numerical value contracts.
2. Harden the expanded UDF surface with examples and broader integration coverage.
3. Push toward publish-ready docs and release metadata.

## V1 Publish Readiness

Not ready.

Blocked by:

1. missing SQL constructors and normalizers
2. missing examples and richer integration coverage
3. missing publish hardening and release-ready documentation
