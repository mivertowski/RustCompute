//! TEE-I/O probe for Blackwell.
//!
//! Phase 0: asserts the Blackwell preset advertises TEE support.
//! Phase 1 (on VM with CC mode enabled): launches a minimal persistent
//! actor and asserts it runs to completion under CC mode.

#![cfg(feature = "cuda")]

use ringkernel_cuda::launch_config::GpuArchitecture;

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

#[test]
fn blackwell_preset_advertises_tee_support() {
    assert!(
        GpuArchitecture::blackwell().supports_tee(),
        "Blackwell preset must report supports_tee = true (TEE-I/O is a Blackwell capability)"
    );
    // Hopper has TEE too (H100 TEE). Both must report true; the
    // distinction is the *I/O* generalization on Blackwell, which is
    // not captured by the compile-time query.
    assert!(GpuArchitecture::hopper().supports_tee());
}

#[test]
#[ignore = "Phase 1 / on-VM: requires a B200 instance with CC mode enabled (often a reboot)"]
fn blackwell_persistent_actor_runs_under_cc_mode() {
    if !is_blackwell() {
        println!("Skipping: not a Blackwell GPU");
        return;
    }
    // TODO(on-VM): launch a trivial persistent actor (one block, one
    // thread, one iteration that writes a sentinel to mapped memory),
    // verify it completes, and verify nvidia-smi reports CC mode
    // active during the run.
    println!("TODO(on-VM): exercise persistent actor under CC mode");
}
