//! Compile-fail tests for ringkernel-derive proc macros.
//!
//! Uses `trybuild` to verify that misuse of derive macros produces
//! clear compile-time errors rather than silently generating wrong code.
//!
//! # CI Behavior
//!
//! Compile error messages vary between Rust versions. These tests are
//! excluded from CI (gated on `TRYBUILD_RUN=1` env var) since the stderr
//! reference files are generated against a specific toolchain version.
//! Run locally with: `TRYBUILD_RUN=1 cargo test -p ringkernel-derive compile_fail`
//! Regenerate stderr with: `TRYBUILD=overwrite cargo test -p ringkernel-derive compile_fail`

#[test]
fn compile_fail() {
    // Skip on CI / cross-version runs where stderr format may differ
    if std::env::var("TRYBUILD_RUN").is_err() {
        eprintln!("Skipping compile_fail tests (set TRYBUILD_RUN=1 to run)");
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
