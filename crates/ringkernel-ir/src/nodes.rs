//! IR node definitions.
//!
//! Defines all operations that can appear in the IR.

use crate::{BlockId, Dimension, IrType, ValueId};

/// An IR instruction that produces a value.
#[derive(Debug, Clone)]
pub struct Instruction {
    /// The value produced by this instruction.
    pub result: ValueId,
    /// The result type.
    pub result_type: IrType,
    /// The operation.
    pub node: IrNode,
}

impl Instruction {
    /// Create a new instruction.
    pub fn new(result: ValueId, result_type: IrType, node: IrNode) -> Self {
        Self {
            result,
            result_type,
            node,
        }
    }
}

/// IR node representing an operation.
#[derive(Debug, Clone)]
pub enum IrNode {
    // ========================================================================
    // Constants and Parameters
    // ========================================================================
    /// Constant value.
    Constant(ConstantValue),
    /// Parameter reference.
    Parameter(usize),
    /// Undefined value (for phi nodes without all predecessors).
    Undef,

    // ========================================================================
    // Binary Operations
    // ========================================================================
    /// Binary operation.
    BinaryOp(BinaryOp, ValueId, ValueId),

    // ========================================================================
    // Unary Operations
    // ========================================================================
    /// Unary operation.
    UnaryOp(UnaryOp, ValueId),

    // ========================================================================
    // Comparison Operations
    // ========================================================================
    /// Comparison operation.
    Compare(CompareOp, ValueId, ValueId),

    // ========================================================================
    // Type Conversions
    // ========================================================================
    /// Cast to a different type.
    Cast(CastKind, ValueId, IrType),

    // ========================================================================
    // Memory Operations
    // ========================================================================
    /// Load from pointer.
    Load(ValueId),
    /// Store to pointer (no result value).
    Store(ValueId, ValueId),
    /// Get element pointer.
    GetElementPtr(ValueId, Vec<ValueId>),
    /// Allocate local variable.
    Alloca(IrType),
    /// Allocate shared memory.
    SharedAlloc(IrType, usize),
    /// Extract struct field.
    ExtractField(ValueId, usize),
    /// Insert struct field.
    InsertField(ValueId, usize, ValueId),

    // ========================================================================
    // GPU Index Operations
    // ========================================================================
    /// Get thread ID.
    ThreadId(Dimension),
    /// Get block ID.
    BlockId(Dimension),
    /// Get block dimension.
    BlockDim(Dimension),
    /// Get grid dimension.
    GridDim(Dimension),
    /// Get global thread ID (block_id * block_dim + thread_id).
    GlobalThreadId(Dimension),
    /// Get warp/wavefront ID.
    WarpId,
    /// Get lane ID within warp.
    LaneId,

    // ========================================================================
    // Synchronization Operations
    // ========================================================================
    /// Threadgroup/block barrier.
    Barrier,
    /// Memory fence.
    MemoryFence(MemoryScope),
    /// Grid-wide sync (cooperative groups).
    GridSync,

    // ========================================================================
    // Atomic Operations
    // ========================================================================
    /// Atomic operation.
    Atomic(AtomicOp, ValueId, ValueId),
    /// Atomic compare-and-swap.
    AtomicCas(ValueId, ValueId, ValueId),

    // ========================================================================
    // Warp/Subgroup Operations
    // ========================================================================
    /// Warp vote (all, any, ballot).
    WarpVote(WarpVoteOp, ValueId),
    /// Warp shuffle.
    WarpShuffle(WarpShuffleOp, ValueId, ValueId),
    /// Warp reduce.
    WarpReduce(WarpReduceOp, ValueId),

    // ========================================================================
    // Math Operations
    // ========================================================================
    /// Math function.
    Math(MathOp, Vec<ValueId>),

    // ========================================================================
    // Control Flow (non-terminator)
    // ========================================================================
    /// Select (ternary operator).
    Select(ValueId, ValueId, ValueId),
    /// Phi node for SSA.
    Phi(Vec<(BlockId, ValueId)>),

    // ========================================================================
    // RingKernel Messaging
    // ========================================================================
    /// Enqueue to output queue.
    K2HEnqueue(ValueId),
    /// Dequeue from input queue.
    H2KDequeue,
    /// Check if input queue is empty.
    H2KIsEmpty,
    /// Send K2K message.
    K2KSend(ValueId, ValueId),
    /// Receive K2K message.
    K2KRecv,
    /// Try receive K2K message (non-blocking).
    K2KTryRecv,

    // ========================================================================
    // HLC Operations
    // ========================================================================
    /// Get current HLC time.
    HlcNow,
    /// Tick HLC.
    HlcTick,
    /// Update HLC from incoming timestamp.
    HlcUpdate(ValueId),

    // ========================================================================
    // Function Call
    // ========================================================================
    /// Call a function.
    Call(String, Vec<ValueId>),
}

/// Constant values.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    /// Boolean constant.
    Bool(bool),
    /// 32-bit signed integer.
    I32(i32),
    /// 64-bit signed integer.
    I64(i64),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Null pointer.
    Null,
    /// Array of constants.
    Array(Vec<ConstantValue>),
    /// Struct constant.
    Struct(Vec<ConstantValue>),
}

impl ConstantValue {
    /// Get the IR type of this constant.
    pub fn ir_type(&self) -> IrType {
        match self {
            ConstantValue::Bool(_) => IrType::BOOL,
            ConstantValue::I32(_) => IrType::I32,
            ConstantValue::I64(_) => IrType::I64,
            ConstantValue::U32(_) => IrType::U32,
            ConstantValue::U64(_) => IrType::U64,
            ConstantValue::F32(_) => IrType::F32,
            ConstantValue::F64(_) => IrType::F64,
            ConstantValue::Null => IrType::ptr(IrType::Void),
            ConstantValue::Array(elements) => {
                if elements.is_empty() {
                    IrType::array(IrType::Void, 0)
                } else {
                    IrType::array(elements[0].ir_type(), elements.len())
                }
            }
            ConstantValue::Struct(_) => IrType::Void, // Would need struct type info
        }
    }
}

/// Binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Remainder/modulo.
    Rem,

    // Bitwise
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Left shift.
    Shl,
    /// Logical right shift.
    Shr,
    /// Arithmetic right shift.
    Sar,

    // Floating-point specific
    /// Fused multiply-add.
    Fma,
    /// Power.
    Pow,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "add"),
            BinaryOp::Sub => write!(f, "sub"),
            BinaryOp::Mul => write!(f, "mul"),
            BinaryOp::Div => write!(f, "div"),
            BinaryOp::Rem => write!(f, "rem"),
            BinaryOp::And => write!(f, "and"),
            BinaryOp::Or => write!(f, "or"),
            BinaryOp::Xor => write!(f, "xor"),
            BinaryOp::Shl => write!(f, "shl"),
            BinaryOp::Shr => write!(f, "shr"),
            BinaryOp::Sar => write!(f, "sar"),
            BinaryOp::Fma => write!(f, "fma"),
            BinaryOp::Pow => write!(f, "pow"),
            BinaryOp::Min => write!(f, "min"),
            BinaryOp::Max => write!(f, "max"),
        }
    }
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negation.
    Neg,
    /// Bitwise NOT.
    Not,
    /// Logical NOT (for booleans).
    LogicalNot,
    /// Absolute value.
    Abs,
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    Rsqrt,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round to nearest.
    Round,
    /// Truncate.
    Trunc,
    /// Sign.
    Sign,
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "neg"),
            UnaryOp::Not => write!(f, "not"),
            UnaryOp::LogicalNot => write!(f, "lnot"),
            UnaryOp::Abs => write!(f, "abs"),
            UnaryOp::Sqrt => write!(f, "sqrt"),
            UnaryOp::Rsqrt => write!(f, "rsqrt"),
            UnaryOp::Floor => write!(f, "floor"),
            UnaryOp::Ceil => write!(f, "ceil"),
            UnaryOp::Round => write!(f, "round"),
            UnaryOp::Trunc => write!(f, "trunc"),
            UnaryOp::Sign => write!(f, "sign"),
        }
    }
}

/// Comparison operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

impl std::fmt::Display for CompareOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareOp::Eq => write!(f, "eq"),
            CompareOp::Ne => write!(f, "ne"),
            CompareOp::Lt => write!(f, "lt"),
            CompareOp::Le => write!(f, "le"),
            CompareOp::Gt => write!(f, "gt"),
            CompareOp::Ge => write!(f, "ge"),
        }
    }
}

/// Cast kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// Bitcast (same size, different type).
    Bitcast,
    /// Zero extend.
    ZeroExtend,
    /// Sign extend.
    SignExtend,
    /// Truncate.
    Truncate,
    /// Float to int.
    FloatToInt,
    /// Int to float.
    IntToFloat,
    /// Float to float (change precision).
    FloatConvert,
    /// Pointer cast.
    PtrCast,
}

/// Memory scope for fences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Thread-local scope.
    Thread,
    /// Threadgroup/block scope.
    Threadgroup,
    /// Device scope.
    Device,
    /// System scope.
    System,
}

/// Atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    /// Atomic load.
    Load,
    /// Atomic store.
    Store,
    /// Atomic exchange.
    Exchange,
    /// Atomic add.
    Add,
    /// Atomic sub.
    Sub,
    /// Atomic min.
    Min,
    /// Atomic max.
    Max,
    /// Atomic AND.
    And,
    /// Atomic OR.
    Or,
    /// Atomic XOR.
    Xor,
}

/// Warp vote operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpVoteOp {
    /// All threads have true.
    All,
    /// Any thread has true.
    Any,
    /// Ballot (bitmask of predicates).
    Ballot,
}

/// Warp shuffle operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpShuffleOp {
    /// Shuffle indexed.
    Index,
    /// Shuffle up.
    Up,
    /// Shuffle down.
    Down,
    /// Shuffle XOR.
    Xor,
}

/// Warp reduce operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpReduceOp {
    /// Sum reduction.
    Sum,
    /// Product reduction.
    Product,
    /// Minimum reduction.
    Min,
    /// Maximum reduction.
    Max,
    /// AND reduction.
    And,
    /// OR reduction.
    Or,
    /// XOR reduction.
    Xor,
}

/// Math operations (intrinsics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathOp {
    // Trigonometric
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Arc sine.
    Asin,
    /// Arc cosine.
    Acos,
    /// Arc tangent.
    Atan,
    /// Arc tangent with two arguments.
    Atan2,

    // Hyperbolic
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,

    // Exponential/Logarithmic
    /// Exponential (e^x).
    Exp,
    /// Exponential base 2.
    Exp2,
    /// Natural logarithm.
    Log,
    /// Logarithm base 2.
    Log2,
    /// Logarithm base 10.
    Log10,

    // Other
    /// Linear interpolation.
    Lerp,
    /// Clamp.
    Clamp,
    /// Step function.
    Step,
    /// Smooth step.
    SmoothStep,
    /// Fract (fractional part).
    Fract,
    /// Copy sign.
    CopySign,
}

/// Block terminator instructions.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return from kernel.
    Return(Option<ValueId>),
    /// Unconditional branch.
    Branch(BlockId),
    /// Conditional branch.
    CondBranch(ValueId, BlockId, BlockId),
    /// Switch statement.
    Switch(ValueId, BlockId, Vec<(ConstantValue, BlockId)>),
    /// Unreachable (for optimization).
    Unreachable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_ir_type() {
        assert_eq!(ConstantValue::I32(42).ir_type(), IrType::I32);
        assert_eq!(ConstantValue::F32(3.14).ir_type(), IrType::F32);
        assert_eq!(ConstantValue::Bool(true).ir_type(), IrType::BOOL);
    }

    #[test]
    fn test_binary_op_display() {
        assert_eq!(format!("{}", BinaryOp::Add), "add");
        assert_eq!(format!("{}", BinaryOp::Mul), "mul");
    }

    #[test]
    fn test_unary_op_display() {
        assert_eq!(format!("{}", UnaryOp::Neg), "neg");
        assert_eq!(format!("{}", UnaryOp::Sqrt), "sqrt");
    }

    #[test]
    fn test_compare_op_display() {
        assert_eq!(format!("{}", CompareOp::Eq), "eq");
        assert_eq!(format!("{}", CompareOp::Lt), "lt");
    }
}
