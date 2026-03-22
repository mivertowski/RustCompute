/// ControlBlockState must fit in 24 bytes (the ControlBlock._reserved field).
/// A 32-byte struct should fail at compile time.
use ringkernel_derive::ControlBlockState;

#[derive(ControlBlockState, Default, Clone, Copy, Debug)]
#[repr(C)]
struct TooLargeState {
    data: [u8; 32], // 32 bytes > 24-byte limit
}

fn main() {}
