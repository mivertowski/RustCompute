//! Compile-fail tests for ringkernel-derive proc macros.
//!
//! Uses `trybuild` to verify that misuse of derive macros produces
//! clear compile-time errors rather than silently generating wrong code.

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
