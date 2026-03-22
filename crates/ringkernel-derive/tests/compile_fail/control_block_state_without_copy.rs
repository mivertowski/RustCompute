/// ControlBlockState requires Copy. A struct containing a Vec cannot be Copy.
use ringkernel_derive::ControlBlockState;

#[derive(ControlBlockState, Default, Clone, Debug)]
#[repr(C)]
struct NotCopyableState {
    data: Vec<u8>,
}

fn main() {}
