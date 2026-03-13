LOG := env('RUST_LOG', '')
provider_features := env('NDATAFUSION_PROVIDER_FEATURES', 'openblas-system')
provider_env_prefix := if os() == "macos" { "env PKG_CONFIG_PATH=/opt/homebrew/opt/openblas/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}} OPENBLAS_DIR=/opt/homebrew/opt/openblas" } else { "env" }

default:
    @just --list

# --- TESTS ---

test-unit:
    RUST_LOG={{ LOG }} cargo test --no-default-features --lib -- --nocapture --show-output

# Runs unit tests first then integration
test:
    RUST_LOG={{ LOG }} cargo test --no-default-features --lib -- --nocapture --show-output
    RUST_LOG={{ LOG }} cargo test --no-default-features --test "e2e" -- --nocapture --show-output

test-one test_name:
    RUST_LOG={{ LOG }} cargo test "{{ test_name }}" -- --nocapture --show-output

test-integration test_name='':
    RUST_LOG={{ LOG }} cargo test --test "e2e" "{{ test_name }}" -- --nocapture --show-output

# --- COVERAGE ---

coverage:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --no-default-features --no-report --ignore-filename-regex "(examples).*"
    cargo llvm-cov report -vv --html --output-dir coverage --open

coverage-lcov:
    cargo llvm-cov clean --workspace
    cargo llvm-cov --no-default-features --lcov --no-report --ignore-filename-regex "(examples).*"
    cargo llvm-cov report --lcov --output-path lcov.info

# --- EXAMPLES ---

example example:
    cargo run --example "{{ example }}"

# --- DOCS ---

docs:
    cargo doc --open

# --- MAINTENANCE ---

fmt:
    cargo +nightly fmt -- --config-path ./rustfmt.toml

fmt-check:
    cargo +nightly fmt --check -- --config-path ./rustfmt.toml

# Run checks CI will
checks:
    cargo +nightly fmt --all -- --check --config-path ./rustfmt.toml
    cargo +nightly clippy --workspace --no-default-features --all-targets -- -D warnings
    cargo +nightly clippy --workspace --no-default-features --features lapack-provider --all-targets -- -D warnings
    cargo +nightly clippy --workspace --no-default-features --features accelerator-rayon --all-targets -- -D warnings
    cargo +nightly clippy --workspace --no-default-features --features accelerator-wgpu --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +nightly clippy --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features lapack-provider --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features accelerator-rayon --all-targets -- -D warnings
    cargo +stable clippy --workspace --no-default-features --features accelerator-wgpu --all-targets -- -D warnings
    {{ provider_env_prefix }} cargo +stable clippy --workspace --no-default-features --features "{{ provider_features }} accelerator-rayon accelerator-wgpu" --all-targets -- -D warnings
    just -f {{ justfile() }} test

# Initialize development environment for maintainers
init-dev:
    @echo "Installing development tools..."
    cargo install cargo-release || true
    cargo install git-cliff || true
    cargo install cargo-edit || true
    cargo install cargo-outdated || true
    cargo install cargo-audit || true
    @echo ""
    @echo "✅ Development tools installed!"
    @echo ""
    @echo "Next steps:"
    @echo "1. Get your crates.io API token from https://crates.io/settings/tokens"
    @echo "2. Add it as CARGO_REGISTRY_TOKEN in GitHub repo settings → Secrets"
    @echo "3. Use 'cargo release patch/minor/major' to create releases"
    @echo ""
    @echo "Useful commands:"
    @echo "  just release-dry patch  # Preview what would happen"
    @echo "  just check-outdated     # Check for outdated dependencies"
    @echo "  just audit              # Security audit"

# Check for outdated dependencies
check-outdated:
    cargo outdated

# Run security audit
audit:
    cargo audit

# Prepare a release (creates PR with version bumps and changelog)
prepare-release version:
    #!/usr/bin/env bash
    set -euo pipefail

    # Validate version format
    if ! [[ "{{ version }}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Version must be in format X.Y.Z (e.g., 0.2.0)"
        exit 1
    fi

    # Parse version components
    IFS='.' read -r MAJOR MINOR PATCH <<< "{{ version }}"

    # Get current version for release notes
    CURRENT_VERSION=$(grep -E '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

    # Create release branch
    git checkout -b "release-v{{ version }}"

    # Update version in root Cargo.toml (in [package] section)
    # This uses a more specific pattern to only match the version under [package]
    awk '/^\[package\]/ {in_package=1} in_package && /^version = / {gsub(/"[^"]*"/, "\"{{ version }}\""); in_package=0} {print}' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

    # Update gravel version references in README files (if they exist)
    # Look for patterns like: gravel = "0.1.1" or gravel = { version = "0.1.1"
    for readme in README.md; do
        if [ -f "$readme" ]; then
            # Update simple dependency format
            sed -i '' "s/ndatafusion = \"[0-9]*\.[0-9]*\.[0-9]*\"/ndatafusion = \"{{ version }}\"/" "$readme" || true
            # Update version field in dependency table format
            sed -i '' "s/ndatafusion = { version = \"[0-9]*\.[0-9]*\.[0-9]*\"/ndatafusion = { version = \"{{ version }}\"/" "$readme" || true
        fi
    done

    # Update Cargo.lock
    cargo update --workspace

    # Generate full changelog
    echo "Generating changelog..."
    git cliff -o CHANGELOG.md

    # Generate release notes for this version
    echo "Generating release notes..."
    git cliff --unreleased --tag v{{ version }} --strip header -o RELEASE_NOTES.md

    # Stage all changes
    git add Cargo.toml Cargo.lock CHANGELOG.md RELEASE_NOTES.md
    # Also add README files if they were modified
    git add README.md 2>/dev/null || true

    # Commit
    git commit -m "chore: prepare release v{{ version }}"

    # Push branch
    git push origin "release-v{{ version }}"

    echo ""
    echo "✅ Release preparation complete!"
    echo ""
    echo "Release notes preview:"
    echo "----------------------"
    head -20 RELEASE_NOTES.md
    echo ""
    echo "Next steps:"
    echo "1. Create a PR from the 'release-v{{ version }}' branch"
    echo "2. Review and merge the PR"
    echo "3. After merge, run: just tag-release {{ version }}"
    echo ""

# Tag a release after the PR is merged
tag-release version:
    #!/usr/bin/env bash
    set -euo pipefail

    # Ensure we're on main and up to date
    git checkout main
    git pull origin main

    # Verify the version in Cargo.toml matches
    CARGO_VERSION=$(grep -E '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    if [ "$CARGO_VERSION" != "{{ version }}" ]; then
        echo "Error: Cargo.toml version ($CARGO_VERSION) does not match requested version ({{ version }})"
        echo "Did the release PR merge successfully?"
        exit 1
    fi

    # Verify publish will work
    cargo publish --dry-run -p ndatafusion --no-verify

    # Create and push tag
    git tag -a "v{{ version }}" -m "Release v{{ version }}"
    git push origin "v{{ version }}"

    echo ""
    echo "✅ Tag v{{ version }} created and pushed!"
    echo "The release workflow will now run automatically."
    echo ""

# Preview what a release would do (dry run)
release-dry version:
    @echo "This would:"
    @echo "1. Create branch: release-v{{ version }}"
    @echo "2. Update version to {{ version }} in:"
    @echo "   - Cargo.toml (workspace.package section only)"
    @echo "   - README files (if they contain gravel version references)"
    @echo "3. Update Cargo.lock (usually done automatically with Cargo.toml change)"
    @echo "4. Generate CHANGELOG.md"
    @echo "5. Generate RELEASE_NOTES.md"
    @echo "6. Create commit and push branch"
    @echo ""
    @echo "After PR merge, 'just tag-release {{ version }}' would:"
    @echo "1. Tag the merged commit as v{{ version }}"
    @echo "2. Push the tag (triggering release workflow)"
