/// RingMessage should only work on structs with named fields, not tuple structs.
use ringkernel_derive::RingMessage;

#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct TupleMessage(u32, f32);

fn main() {}
