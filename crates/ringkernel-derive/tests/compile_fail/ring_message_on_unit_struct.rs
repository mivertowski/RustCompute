/// RingMessage should only work on structs with named fields, not unit structs.
use ringkernel_derive::RingMessage;

#[derive(Debug, Clone, RingMessage)]
struct UnitMessage;

fn main() {}
