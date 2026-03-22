/// PersistentMessage requires a handler_id attribute.
/// Omitting it should produce a compile error.
use ringkernel_derive::{PersistentMessage, RingMessage};

#[derive(
    RingMessage,
    PersistentMessage,
    Clone,
    Copy,
    Debug,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[archive(check_bytes)]
#[repr(C)]
#[message(type_id = 9999)]
// Missing: #[persistent_message(handler_id = ...)]
struct MissingHandlerId {
    value: u32,
}

fn main() {}
