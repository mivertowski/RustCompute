/// RingMessage should only work on structs with named fields, not enums.
use ringkernel_derive::RingMessage;

#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
enum NotAStruct {
    Variant1,
    Variant2(u32),
}

fn main() {}
