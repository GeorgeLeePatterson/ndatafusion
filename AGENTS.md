# AGENTS.md

Repository-level instructions for human and LLM contributors.

## Scope

Applies to the entire repository.

## Mission

Build `ndatafusion` into a production-grade DataFusion extension crate that exposes `nabled`
capabilities through explicit Arrow/DataFusion contracts without forking numerical semantics from
`nabled`.

## Mandatory Context Bootstrap

Before making architectural or broad refactor changes, read these in order:

1. `docs/README.md`
2. `docs/DECISIONS.md`
3. `docs/CAPABILITY_MATRIX.md`
4. `docs/EXECUTION_TRACKER.md`
5. `docs/PUBLISH_CHECKLIST.md` when release, docs.rs, or publish posture is relevant
6. `docs/ARCHITECTURE.md`
7. `docs/STATUS.md`

Do not infer status from memory. Use:

1. `docs/STATUS.md` for current project state.
2. `docs/CAPABILITY_MATRIX.md` for scope gaps and v1 sufficiency.
3. `docs/EXECUTION_TRACKER.md` for `Done / Next / Needed` execution state.
4. Resume from the highest-priority open `N-*` item in `docs/EXECUTION_TRACKER.md` unless
   explicitly redirected.
5. If the highest-priority open `N-*` item is an upstream prerequisite in `../ndarrow` or
   `../nabled`, complete that work first before resuming local `ndatafusion` implementation.

## Non-Negotiable Constraints

1. `ndatafusion` is a DataFusion-facing facade over `nabled`; it does not own new numerical
   kernels.
2. `nabled` remains the source of truth for algorithm semantics, provider/backend/kernel routing,
   and Arrow boundary assumptions.
3. Capability parity matters more than 1:1 overload mirroring.
4. Admit only natural Arrow/DataFusion contracts with explicit shape and result semantics.
5. No hidden copy-heavy conversions in hot paths.
6. Prefer zero-copy row extraction and explicit materialization only when the output contract
   requires it.
7. Preserve `nabled` feature forwarding exactly, except `nabled/arrow`, which is part of the base
   `ndatafusion` contract and is enabled unconditionally.
8. Keep execution-axis terminology explicit and consistent:
   - `Provider` = decomposition implementation source
   - `Backend` = primitive-kernel execution target
   - `Kernel` = operation-family backend contract
9. Use the new module layout only. Do not create `mod.rs`.
10. Prefer direct batch-native delegation into stabilized lower-layer Arrow APIs. Treat row/cell
    codecs as fallback-only when the lower layers do not yet admit the needed contract.
11. Cross-repo prerequisite work must not regress correctness or performance in already admitted
    lower-layer public behavior.
12. Do not narrow or remove broader lower-layer public APIs solely because `ndatafusion` chooses a
    stricter SQL-facing contract surface.

## Repository Rules

1. Treat `.justfile` as the canonical task runner.
2. Run `just checks` after every legitimate checkpoint.
3. `just checks` is not optional. It runs:
   - `cargo +nightly fmt --all -- --check --config-path ./rustfmt.toml`
   - nightly/stable clippy over the supported `no-default-features` feature matrix, including
     provider-gated and accelerator-gated combinations
   - all unit and integration tests via `just test`
4. Keep the `e2e` integration test target present, because `.justfile` expects
   `cargo test --test e2e`.
5. Before any commit is pushed, line coverage must be greater than 90%. Use `cargo llvm-cov` via
   the coverage commands in `.justfile` to verify this.
6. Clippy pedantic lints are enabled in `Cargo.toml` and must be addressed.
7. Do not silence pedantic lints with `allow` unless there is no viable alternative.
8. Prefer fixing the underlying code. If a conscious assertion is needed, `expect` is preferable
   to `allow`.
9. Any unavoidable lint exception must be narrowly scoped and justified.
10. The guiding principle for all design and implementation work is:
    - `algebraic, homomorphic, compositional built from denotational semantics`

## Quality Gates

Run and pass before finalizing:

1. `just checks`

Coverage expectation:

1. Keep line coverage greater than 90% as a hard gate for merge/push.
2. If coverage drops below this threshold, add meaningful tests before finalizing.
3. Prefer meaningful coverage over synthetic assertions.

## Test Placement

1. Unit tests in module-local `#[cfg(test)]` blocks.
2. `tests/` for cross-module integration/e2e behavior.

## Documentation Discipline

When architecture or behavior changes, update docs in the same change set:

1. `README.md` for user-visible behavior.
2. `docs/STATUS.md` for migration and implementation truth.
3. `docs/EXECUTION_TRACKER.md` for execution progression (`Done / Next / Needed`).
4. `docs/CAPABILITY_MATRIX.md` for scope and sufficiency updates.
5. `docs/ARCHITECTURE.md` if the target design or sequencing changes.
6. `docs/DECISIONS.md` when assumptions or constraints are locked or changed.
7. `docs/PUBLISH_CHECKLIST.md` when release posture, docs.rs assumptions, or publish blockers
   change.
8. Relevant rustdoc comments for API contract changes.

## Reference Sources

1. Use `../ndarrow` as the primary Arrow/ndarray contract reference.
2. Use `../nabled` as the primary numerical semantics and Arrow façade reference.
3. Use `../../packages/datafusion` to validate DataFusion interfaces and compatibility details.
4. Use `../../packages/datafusion-functions-json` as a reference for standalone DataFusion function
   library structure.
5. When dependency work touches Arrow compatibility, verify `datafusion`, `ndarrow`, and `nabled` Arrow
   versions explicitly before changing versions.
