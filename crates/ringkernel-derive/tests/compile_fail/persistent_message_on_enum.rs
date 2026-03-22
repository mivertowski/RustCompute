/// PersistentMessage should only work on structs with named fields, not enums.
use ringkernel_derive::{PersistentMessage, RingMessage};

#[derive(RingMessage, PersistentMessage, Clone, Copy, Debug)]
#[persistent_message(handler_id = 1)]
enum NotAStruct {
    A,
    B,
}

fn main() {}
