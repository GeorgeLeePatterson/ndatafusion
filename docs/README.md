# ndatafusion Docs

This folder contains the current, authoritative direction for `ndatafusion`.

## Current Direction

1. `ndatafusion` is a DataFusion extension crate over `nabled`.
2. `nabled` remains the canonical owner of numerical semantics, provider/backend/kernel routing,
   and admitted Arrow boundary shapes.
3. `ndarrow` remains the canonical Arrow/ndarray boundary layer.
4. `ndatafusion` owns the DataFusion-specific registration surface and SQL-facing contracts.
5. `ndatafusion` should prefer direct batch-native delegation into `nabled::arrow` when the lower
   layers admit the relevant shape efficiently.
6. A DataFusion cell-codec layer still exists, but only as a fallback for admitted workflows that
   do not yet have a direct batch-native lower-layer contract.
7. Upstream correctness and performance must not regress while closing these gaps.
8. Existing broader `nabled::arrow` public APIs may remain when they are useful to non-`ndatafusion`
   consumers; `ndatafusion` does not need to narrow them just because it uses stricter contracts.
9. The canonical design matrix is concept-first: each mathematical object family should own one
   canonical standalone ingress and one canonical `rows-of-X` batch carrier.
10. Upstream work is gated by two release checkpoints:
    - checkpoint 1: `ndarrow` is ready for release
    - checkpoint 2: `nabled` is ready for release
11. Those checkpoints are now complete and published:
    - `ndarrow 0.0.3`
    - `nabled 0.0.7`
12. Capability parity matters more than 1:1 overload mirroring.
13. Copy-light Arrow/DataFusion execution is mandatory; heavy hidden materialization is not.
14. V1 publish scope is UDF-first. UDAFs, window functions, table functions, and planner sugar are
    optional only when they clearly improve a natural SQL contract.

## Documents

1. `docs/DECISIONS.md`: locked architectural decisions and constraints.
2. `docs/CAPABILITY_MATRIX.md`: capability inventory, gap map, and v1 sufficiency verdict.
3. `docs/EXECUTION_TRACKER.md`: canonical `Done / Next / Needed` tracker for continuation.
4. `docs/PUBLISH_CHECKLIST.md`: release gate, docs.rs posture, release-note minimums, and current
   publication blockers.
5. `docs/ARCHITECTURE.md`: end-to-end implementation plan from scaffold to v1 publish.
6. `docs/STATUS.md`: current repository snapshot and next required milestone.

## Context Resume Protocol

When starting from a compacted or partial context, read documents in this order:

1. `docs/README.md`
2. `docs/DECISIONS.md`
3. `docs/CAPABILITY_MATRIX.md`
4. `docs/EXECUTION_TRACKER.md`
5. `docs/PUBLISH_CHECKLIST.md` when release, docs.rs, or publish posture is relevant
6. `docs/ARCHITECTURE.md`
7. `docs/STATUS.md`

Use `docs/EXECUTION_TRACKER.md` to resume from `Next` items first. Avoid a full repository
re-assessment unless tracker state is stale or contradictory.

Then verify repository state quickly:

1. `cargo metadata --no-deps`
2. `rg --files`
3. `sed -n '1,220p' Cargo.toml`

## Context Sufficiency Check

After reading the docs above, a contributor should be able to answer:

1. What is `ndatafusion`'s mission? (Expose `nabled` through DataFusion-native function contracts.)
2. What does `ndatafusion` own vs `ndarrow` and `nabled`? (`ndarrow` owns Arrow/ndarray bridge
   contracts, `nabled` owns numerical semantics and batch/object Arrow contracts, and `ndatafusion`
   owns DataFusion registration plus any fallback lift/assembly logic.)
3. What is the canonical progress tracker? (`docs/EXECUTION_TRACKER.md`)
4. What is the current v1 sufficiency verdict? (`docs/CAPABILITY_MATRIX.md`)
5. What is the next milestone? (`docs/STATUS.md` and `docs/EXECUTION_TRACKER.md`)
6. What are the mandatory quality gates? (`just checks`, coverage greater than 90% before push)
7. Which DataFusion extension seams are in play? (`ScalarUDF` first, with optional planners or
   rewrites only when needed)
8. What boundary shapes are currently admitted? (Batch-native and fallback-lifted forms of
   `nabled`'s vector, matrix, tensor, sparse, and complex Arrow contracts, with the required
   upstream hardening now released in `ndarrow` and `nabled`)

## Scope Boundary

1. `ndatafusion` does not implement numerical kernels that diverge from `nabled`.
2. This crate may add DataFusion-specific planners, codecs, and constructors, but those must still
   delegate to canonical `nabled` behavior and should not compensate for avoidable lower-layer
   contract or performance gaps.
3. Changes that alter admitted Arrow/DataFusion contracts must update `docs/DECISIONS.md`,
   `docs/CAPABILITY_MATRIX.md`, and `docs/STATUS.md` in the same change set.
