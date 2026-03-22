/// GpuType requires the type to implement Copy.
/// A struct with a String field cannot be Copy.
use ringkernel_derive::GpuType;

#[derive(Debug, Clone, GpuType)]
#[repr(C)]
struct NotCopyable {
    name: String,
}

fn main() {}
