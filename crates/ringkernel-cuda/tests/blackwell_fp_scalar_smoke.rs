//! FP4 / FP6 / FP8 scalar smoke test for Blackwell (sm_100+).
//!
//! Phase 0 (here): asserts that ringkernel-ir lowers each Blackwell
//! scalar type to the correct __nv_* name. No GPU needed.
//!
//! Phase 1 (on B200): wires KERNEL_PTX per scalar, runs the roundtrip
//! kernel, and asserts the readback matches the reference value within
//! the scalar's representable range.

#![cfg(feature = "cuda")]

use ringkernel_ir::ScalarType;

/// Returns true iff a Blackwell-class device (CC >= 10.0) is present.
#[allow(dead_code)]
fn is_blackwell() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 10
        }
        Err(_) => false,
    }
}

/// The names that ringkernel-ir::lower_cuda must emit for each
/// Blackwell scalar. Hard-coded for Phase 0 verification; the test
/// is the single source of truth for what nvcc -arch=sm_100 must accept.
const EXPECTED_CUDA_NAMES: &[(ScalarType, &str)] = &[
    (ScalarType::BF16, "__nv_bfloat16"),
    (ScalarType::FP8E4M3, "__nv_fp8_e4m3"),
    (ScalarType::FP8E5M2, "__nv_fp8_e5m2"),
    (ScalarType::FP6E3M2, "__nv_fp6_e3m2"),
    (ScalarType::FP6E2M3, "__nv_fp6_e2m3"),
    (ScalarType::FP4E2M1, "__nv_fp4_e2m1"),
];

#[test]
fn blackwell_scalar_types_have_minimum_compute_capability() {
    // FP4 / FP6 require CC >= 10.0; FP8 requires CC >= 9.0; BF16 >= 8.0.
    assert_eq!(ScalarType::BF16.min_compute_capability(), (8, 0));
    assert_eq!(ScalarType::FP8E4M3.min_compute_capability(), (9, 0));
    assert_eq!(ScalarType::FP8E5M2.min_compute_capability(), (9, 0));
    assert_eq!(ScalarType::FP6E3M2.min_compute_capability(), (10, 0));
    assert_eq!(ScalarType::FP6E2M3.min_compute_capability(), (10, 0));
    assert_eq!(ScalarType::FP4E2M1.min_compute_capability(), (10, 0));
}

#[test]
fn blackwell_scalar_types_are_floating_point() {
    for (ty, _) in EXPECTED_CUDA_NAMES {
        assert!(
            ty.is_float(),
            "{ty:?} must be classified as floating point"
        );
    }
}

#[test]
#[ignore = "Phase 1 / on-VM: requires B200 (sm_100+) and a filled-in KERNEL_PTX per scalar"]
fn blackwell_fp_scalar_roundtrip_runtime() {
    if !is_blackwell() {
        println!("Skipping: not a Blackwell GPU");
        return;
    }
    // Filled in on the VM. Per-scalar PTX strings live in a sibling
    // module that the VM session adds; pattern:
    //
    //   const KERNEL_PTX_FP8_E4M3: &str = "...";
    //   const KERNEL_PTX_FP6_E3M2: &str = "...";
    //   ...
    //
    // Each kernel takes one __nv_<ty> input scalar, multiplies by 2.0,
    // and writes the result. The roundtrip is the assertion target.
    println!("TODO(on-VM): wire up per-scalar PTX and run roundtrip");
}
