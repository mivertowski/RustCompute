//! Optimization passes for the IR.
//!
//! This module provides optimization passes that transform IR modules:
//!
//! - **Dead Code Elimination (DCE)**: Remove instructions whose results are never used
//! - **Constant Folding**: Evaluate operations on constants at compile time
//! - **Constant Propagation**: Replace uses of constants with their values
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ir::{IrModule, optimize};
//!
//! let module = build_ir();
//! let optimized = optimize::run_all_passes(&module);
//! ```

use std::collections::{HashMap, HashSet};

use crate::{
    BinaryOp, BlockId, CompareOp, ConstantValue, IrModule, IrNode, IrType, ScalarType, Terminator,
    UnaryOp, ValueId,
};

// ============================================================================
// OPTIMIZATION PASS INTERFACE
// ============================================================================

/// An optimization pass that transforms an IR module.
pub trait OptimizationPass {
    /// Run the optimization pass on the module.
    fn run(&self, module: &mut IrModule) -> OptimizationResult;

    /// Get the name of this pass.
    fn name(&self) -> &'static str;
}

/// Result of running an optimization pass.
#[derive(Debug, Clone, Default)]
pub struct OptimizationResult {
    /// Whether the module was changed.
    pub changed: bool,
    /// Number of instructions removed.
    pub instructions_removed: usize,
    /// Number of instructions modified.
    pub instructions_modified: usize,
    /// Number of blocks removed.
    pub blocks_removed: usize,
}

impl OptimizationResult {
    /// Create a result indicating no changes.
    pub fn unchanged() -> Self {
        Self::default()
    }

    /// Create a result indicating changes were made.
    pub fn changed() -> Self {
        Self {
            changed: true,
            ..Default::default()
        }
    }

    /// Merge with another result.
    pub fn merge(&mut self, other: OptimizationResult) {
        self.changed |= other.changed;
        self.instructions_removed += other.instructions_removed;
        self.instructions_modified += other.instructions_modified;
        self.blocks_removed += other.blocks_removed;
    }
}

// ============================================================================
// DEAD CODE ELIMINATION
// ============================================================================

/// Dead Code Elimination pass.
///
/// Removes instructions whose results are never used.
pub struct DeadCodeElimination;

impl DeadCodeElimination {
    /// Create a new DCE pass.
    pub fn new() -> Self {
        Self
    }

    /// Find all values that are used in the module.
    fn find_used_values(&self, module: &IrModule) -> HashSet<ValueId> {
        let mut used = HashSet::new();

        // Parameters are always used
        for param in &module.parameters {
            used.insert(param.value_id);
        }

        // Traverse all blocks
        for block in module.blocks.values() {
            // Collect uses from instructions
            for inst in &block.instructions {
                self.collect_uses(&inst.node, &mut used);
            }

            // Collect uses from terminator
            if let Some(ref term) = block.terminator {
                self.collect_terminator_uses(term, &mut used);
            }
        }

        used
    }

    /// Collect all value uses from a node.
    fn collect_uses(&self, node: &IrNode, used: &mut HashSet<ValueId>) {
        match node {
            IrNode::BinaryOp(_, lhs, rhs) => {
                used.insert(*lhs);
                used.insert(*rhs);
            }
            IrNode::UnaryOp(_, operand) => {
                used.insert(*operand);
            }
            IrNode::Compare(_, lhs, rhs) => {
                used.insert(*lhs);
                used.insert(*rhs);
            }
            IrNode::Cast(_, value, _) => {
                used.insert(*value);
            }
            IrNode::Load(ptr) => {
                used.insert(*ptr);
            }
            IrNode::Store(ptr, value) => {
                used.insert(*ptr);
                used.insert(*value);
            }
            IrNode::GetElementPtr(base, indices) => {
                used.insert(*base);
                for idx in indices {
                    used.insert(*idx);
                }
            }
            IrNode::Select(cond, then_val, else_val) => {
                used.insert(*cond);
                used.insert(*then_val);
                used.insert(*else_val);
            }
            IrNode::Phi(incoming) => {
                for (_, value) in incoming {
                    used.insert(*value);
                }
            }
            IrNode::Atomic(_, ptr, value) => {
                used.insert(*ptr);
                used.insert(*value);
            }
            IrNode::AtomicCas(ptr, expected, desired) => {
                used.insert(*ptr);
                used.insert(*expected);
                used.insert(*desired);
            }
            IrNode::WarpVote(_, pred) => {
                used.insert(*pred);
            }
            IrNode::WarpShuffle(_, value, lane) => {
                used.insert(*value);
                used.insert(*lane);
            }
            IrNode::WarpReduce(_, value) => {
                used.insert(*value);
            }
            IrNode::Math(_, args) => {
                for arg in args {
                    used.insert(*arg);
                }
            }
            IrNode::Call(_, args) => {
                for arg in args {
                    used.insert(*arg);
                }
            }
            IrNode::K2HEnqueue(value) => {
                used.insert(*value);
            }
            IrNode::K2KSend(dest, msg) => {
                used.insert(*dest);
                used.insert(*msg);
            }
            IrNode::HlcUpdate(ts) => {
                used.insert(*ts);
            }
            IrNode::ExtractField(value, _) => {
                used.insert(*value);
            }
            IrNode::InsertField(base, _, value) => {
                used.insert(*base);
                used.insert(*value);
            }
            // No uses for these nodes
            IrNode::Constant(_)
            | IrNode::Parameter(_)
            | IrNode::Undef
            | IrNode::ThreadId(_)
            | IrNode::BlockId(_)
            | IrNode::BlockDim(_)
            | IrNode::GridDim(_)
            | IrNode::GlobalThreadId(_)
            | IrNode::WarpId
            | IrNode::LaneId
            | IrNode::Barrier
            | IrNode::MemoryFence(_)
            | IrNode::GridSync
            | IrNode::Alloca(_)
            | IrNode::SharedAlloc(_, _)
            | IrNode::H2KDequeue
            | IrNode::H2KIsEmpty
            | IrNode::K2KRecv
            | IrNode::K2KTryRecv
            | IrNode::HlcNow
            | IrNode::HlcTick => {}
        }
    }

    /// Collect uses from a terminator.
    fn collect_terminator_uses(&self, term: &Terminator, used: &mut HashSet<ValueId>) {
        match term {
            Terminator::Return(Some(value)) => {
                used.insert(*value);
            }
            Terminator::CondBranch(cond, _, _) => {
                used.insert(*cond);
            }
            Terminator::Switch(value, _, _) => {
                used.insert(*value);
            }
            Terminator::Return(None) | Terminator::Branch(_) | Terminator::Unreachable => {}
        }
    }

    /// Check if an instruction has side effects and cannot be removed.
    fn has_side_effects(&self, node: &IrNode) -> bool {
        matches!(
            node,
            IrNode::Store(_, _)
                | IrNode::Atomic(_, _, _)
                | IrNode::AtomicCas(_, _, _)
                | IrNode::Barrier
                | IrNode::MemoryFence(_)
                | IrNode::GridSync
                | IrNode::Call(_, _)
                | IrNode::K2HEnqueue(_)
                | IrNode::K2KSend(_, _)
                | IrNode::HlcTick
                | IrNode::HlcUpdate(_)
        )
    }
}

impl Default for DeadCodeElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for DeadCodeElimination {
    fn run(&self, module: &mut IrModule) -> OptimizationResult {
        let used = self.find_used_values(module);
        let mut result = OptimizationResult::unchanged();

        // Remove unused instructions from each block
        for block in module.blocks.values_mut() {
            let original_len = block.instructions.len();

            block.instructions.retain(|inst| {
                // Keep if result is used OR has side effects
                used.contains(&inst.result) || self.has_side_effects(&inst.node)
            });

            let removed = original_len - block.instructions.len();
            if removed > 0 {
                result.changed = true;
                result.instructions_removed += removed;
            }
        }

        result
    }

    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }
}

// ============================================================================
// CONSTANT FOLDING
// ============================================================================

/// Constant Folding pass.
///
/// Evaluates operations on constants at compile time.
pub struct ConstantFolding {
    /// Map from value IDs to their constant values (used for incremental folding).
    #[allow(dead_code)]
    constants: HashMap<ValueId, ConstantValue>,
}

impl ConstantFolding {
    /// Create a new constant folding pass.
    pub fn new() -> Self {
        Self {
            constants: HashMap::new(),
        }
    }

    /// Try to fold a binary operation.
    fn fold_binary_op(
        &self,
        op: BinaryOp,
        lhs: &ConstantValue,
        rhs: &ConstantValue,
    ) -> Option<ConstantValue> {
        match (lhs, rhs) {
            (ConstantValue::I32(l), ConstantValue::I32(r)) => {
                Some(ConstantValue::I32(Self::fold_binary_i32(op, *l, *r)?))
            }
            (ConstantValue::U32(l), ConstantValue::U32(r)) => {
                Some(ConstantValue::U32(Self::fold_binary_u32(op, *l, *r)?))
            }
            (ConstantValue::F32(l), ConstantValue::F32(r)) => {
                Some(ConstantValue::F32(Self::fold_binary_f32(op, *l, *r)?))
            }
            (ConstantValue::I64(l), ConstantValue::I64(r)) => {
                Some(ConstantValue::I64(Self::fold_binary_i64(op, *l, *r)?))
            }
            (ConstantValue::U64(l), ConstantValue::U64(r)) => {
                Some(ConstantValue::U64(Self::fold_binary_u64(op, *l, *r)?))
            }
            (ConstantValue::F64(l), ConstantValue::F64(r)) => {
                Some(ConstantValue::F64(Self::fold_binary_f64(op, *l, *r)?))
            }
            _ => None,
        }
    }

    fn fold_binary_i32(op: BinaryOp, l: i32, r: i32) -> Option<i32> {
        Some(match op {
            BinaryOp::Add => l.wrapping_add(r),
            BinaryOp::Sub => l.wrapping_sub(r),
            BinaryOp::Mul => l.wrapping_mul(r),
            BinaryOp::Div => l.checked_div(r)?,
            BinaryOp::Rem => l.checked_rem(r)?,
            BinaryOp::And => l & r,
            BinaryOp::Or => l | r,
            BinaryOp::Xor => l ^ r,
            BinaryOp::Shl => l.wrapping_shl(r as u32),
            BinaryOp::Shr => l.wrapping_shr(r as u32),
            BinaryOp::Sar => l >> (r as u32),
            BinaryOp::Min => l.min(r),
            BinaryOp::Max => l.max(r),
            _ => return None,
        })
    }

    fn fold_binary_u32(op: BinaryOp, l: u32, r: u32) -> Option<u32> {
        Some(match op {
            BinaryOp::Add => l.wrapping_add(r),
            BinaryOp::Sub => l.wrapping_sub(r),
            BinaryOp::Mul => l.wrapping_mul(r),
            BinaryOp::Div => l.checked_div(r)?,
            BinaryOp::Rem => l.checked_rem(r)?,
            BinaryOp::And => l & r,
            BinaryOp::Or => l | r,
            BinaryOp::Xor => l ^ r,
            BinaryOp::Shl => l.wrapping_shl(r),
            BinaryOp::Shr => l.wrapping_shr(r),
            BinaryOp::Sar => l >> r,
            BinaryOp::Min => l.min(r),
            BinaryOp::Max => l.max(r),
            _ => return None,
        })
    }

    fn fold_binary_i64(op: BinaryOp, l: i64, r: i64) -> Option<i64> {
        Some(match op {
            BinaryOp::Add => l.wrapping_add(r),
            BinaryOp::Sub => l.wrapping_sub(r),
            BinaryOp::Mul => l.wrapping_mul(r),
            BinaryOp::Div => l.checked_div(r)?,
            BinaryOp::Rem => l.checked_rem(r)?,
            BinaryOp::And => l & r,
            BinaryOp::Or => l | r,
            BinaryOp::Xor => l ^ r,
            BinaryOp::Shl => l.wrapping_shl(r as u32),
            BinaryOp::Shr => l.wrapping_shr(r as u32),
            BinaryOp::Sar => l >> (r as u32),
            BinaryOp::Min => l.min(r),
            BinaryOp::Max => l.max(r),
            _ => return None,
        })
    }

    fn fold_binary_u64(op: BinaryOp, l: u64, r: u64) -> Option<u64> {
        Some(match op {
            BinaryOp::Add => l.wrapping_add(r),
            BinaryOp::Sub => l.wrapping_sub(r),
            BinaryOp::Mul => l.wrapping_mul(r),
            BinaryOp::Div => l.checked_div(r)?,
            BinaryOp::Rem => l.checked_rem(r)?,
            BinaryOp::And => l & r,
            BinaryOp::Or => l | r,
            BinaryOp::Xor => l ^ r,
            BinaryOp::Shl => l.wrapping_shl(r as u32),
            BinaryOp::Shr => l.wrapping_shr(r as u32),
            BinaryOp::Sar => l >> (r as u32),
            BinaryOp::Min => l.min(r),
            BinaryOp::Max => l.max(r),
            _ => return None,
        })
    }

    fn fold_binary_f32(op: BinaryOp, l: f32, r: f32) -> Option<f32> {
        Some(match op {
            BinaryOp::Add => l + r,
            BinaryOp::Sub => l - r,
            BinaryOp::Mul => l * r,
            BinaryOp::Div => l / r,
            BinaryOp::Rem => l % r,
            BinaryOp::Min => l.min(r),
            BinaryOp::Max => l.max(r),
            BinaryOp::Pow => l.powf(r),
            _ => return None,
        })
    }

    fn fold_binary_f64(op: BinaryOp, l: f64, r: f64) -> Option<f64> {
        Some(match op {
            BinaryOp::Add => l + r,
            BinaryOp::Sub => l - r,
            BinaryOp::Mul => l * r,
            BinaryOp::Div => l / r,
            BinaryOp::Rem => l % r,
            BinaryOp::Min => l.min(r),
            BinaryOp::Max => l.max(r),
            BinaryOp::Pow => l.powf(r),
            _ => return None,
        })
    }

    /// Try to fold a unary operation.
    fn fold_unary_op(&self, op: UnaryOp, operand: &ConstantValue) -> Option<ConstantValue> {
        match operand {
            ConstantValue::I32(v) => Some(ConstantValue::I32(Self::fold_unary_i32(op, *v)?)),
            ConstantValue::U32(v) => Some(ConstantValue::U32(Self::fold_unary_u32(op, *v)?)),
            ConstantValue::F32(v) => Some(ConstantValue::F32(Self::fold_unary_f32(op, *v)?)),
            ConstantValue::F64(v) => Some(ConstantValue::F64(Self::fold_unary_f64(op, *v)?)),
            ConstantValue::Bool(v) => {
                if op == UnaryOp::LogicalNot {
                    Some(ConstantValue::Bool(!v))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn fold_unary_i32(op: UnaryOp, v: i32) -> Option<i32> {
        Some(match op {
            UnaryOp::Neg => -v,
            UnaryOp::Not => !v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sign => v.signum(),
            _ => return None,
        })
    }

    fn fold_unary_u32(op: UnaryOp, v: u32) -> Option<u32> {
        Some(match op {
            UnaryOp::Not => !v,
            _ => return None,
        })
    }

    fn fold_unary_f32(op: UnaryOp, v: f32) -> Option<f32> {
        Some(match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Rsqrt => 1.0 / v.sqrt(),
            UnaryOp::Floor => v.floor(),
            UnaryOp::Ceil => v.ceil(),
            UnaryOp::Round => v.round(),
            UnaryOp::Trunc => v.trunc(),
            UnaryOp::Sign => v.signum(),
            _ => return None,
        })
    }

    fn fold_unary_f64(op: UnaryOp, v: f64) -> Option<f64> {
        Some(match op {
            UnaryOp::Neg => -v,
            UnaryOp::Abs => v.abs(),
            UnaryOp::Sqrt => v.sqrt(),
            UnaryOp::Rsqrt => 1.0 / v.sqrt(),
            UnaryOp::Floor => v.floor(),
            UnaryOp::Ceil => v.ceil(),
            UnaryOp::Round => v.round(),
            UnaryOp::Trunc => v.trunc(),
            UnaryOp::Sign => v.signum(),
            _ => return None,
        })
    }

    /// Try to fold a comparison.
    fn fold_compare(
        &self,
        op: CompareOp,
        lhs: &ConstantValue,
        rhs: &ConstantValue,
    ) -> Option<ConstantValue> {
        let result = match (lhs, rhs) {
            (ConstantValue::I32(l), ConstantValue::I32(r)) => Self::compare_i32(op, *l, *r),
            (ConstantValue::U32(l), ConstantValue::U32(r)) => Self::compare_u32(op, *l, *r),
            (ConstantValue::F32(l), ConstantValue::F32(r)) => Self::compare_f32(op, *l, *r),
            (ConstantValue::Bool(l), ConstantValue::Bool(r)) => match op {
                CompareOp::Eq => *l == *r,
                CompareOp::Ne => *l != *r,
                _ => return None,
            },
            _ => return None,
        };
        Some(ConstantValue::Bool(result))
    }

    fn compare_i32(op: CompareOp, l: i32, r: i32) -> bool {
        match op {
            CompareOp::Eq => l == r,
            CompareOp::Ne => l != r,
            CompareOp::Lt => l < r,
            CompareOp::Le => l <= r,
            CompareOp::Gt => l > r,
            CompareOp::Ge => l >= r,
        }
    }

    fn compare_u32(op: CompareOp, l: u32, r: u32) -> bool {
        match op {
            CompareOp::Eq => l == r,
            CompareOp::Ne => l != r,
            CompareOp::Lt => l < r,
            CompareOp::Le => l <= r,
            CompareOp::Gt => l > r,
            CompareOp::Ge => l >= r,
        }
    }

    fn compare_f32(op: CompareOp, l: f32, r: f32) -> bool {
        match op {
            CompareOp::Eq => l == r,
            CompareOp::Ne => l != r,
            CompareOp::Lt => l < r,
            CompareOp::Le => l <= r,
            CompareOp::Gt => l > r,
            CompareOp::Ge => l >= r,
        }
    }

    /// Get a constant value for a value ID if available (for future use).
    #[allow(dead_code)]
    fn get_constant<'a>(&'a self, id: ValueId, module: &'a IrModule) -> Option<&'a ConstantValue> {
        // First check our map
        if let Some(c) = self.constants.get(&id) {
            return Some(c);
        }

        // Then check if it's defined as a constant in the module
        if let Some(value) = module.get_value(id) {
            if let IrNode::Constant(ref c) = value.node {
                return Some(c);
            }
        }

        None
    }
}

impl Default for ConstantFolding {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ConstantFolding {
    fn run(&self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        let mut constants = HashMap::new();

        // First pass: collect all constants
        for value in module.values.values() {
            if let IrNode::Constant(ref c) = value.node {
                constants.insert(value.id, c.clone());
            }
        }

        // Second pass: fold operations
        for block in module.blocks.values_mut() {
            for inst in &mut block.instructions {
                let folded = match &inst.node {
                    IrNode::BinaryOp(op, lhs, rhs) => {
                        let lhs_const = constants.get(lhs);
                        let rhs_const = constants.get(rhs);

                        if let (Some(l), Some(r)) = (lhs_const, rhs_const) {
                            Self::new().fold_binary_op(*op, l, r)
                        } else {
                            None
                        }
                    }
                    IrNode::UnaryOp(op, operand) => {
                        if let Some(c) = constants.get(operand) {
                            Self::new().fold_unary_op(*op, c)
                        } else {
                            None
                        }
                    }
                    IrNode::Compare(op, lhs, rhs) => {
                        let lhs_const = constants.get(lhs);
                        let rhs_const = constants.get(rhs);

                        if let (Some(l), Some(r)) = (lhs_const, rhs_const) {
                            Self::new().fold_compare(*op, l, r)
                        } else {
                            None
                        }
                    }
                    IrNode::Select(cond, then_val, else_val) => {
                        if let Some(ConstantValue::Bool(c)) = constants.get(cond) {
                            // Fold to one branch
                            let selected = if *c { then_val } else { else_val };
                            constants.get(selected).cloned()
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(constant) = folded {
                    // Replace instruction with constant
                    let new_type = constant.ir_type();
                    inst.node = IrNode::Constant(constant.clone());
                    inst.result_type = new_type;
                    constants.insert(inst.result, constant);
                    result.changed = true;
                    result.instructions_modified += 1;
                }
            }
        }

        result
    }

    fn name(&self) -> &'static str {
        "constant-folding"
    }
}

// ============================================================================
// DEAD BLOCK ELIMINATION
// ============================================================================

/// Dead Block Elimination pass.
///
/// Removes unreachable blocks from the control flow graph.
pub struct DeadBlockElimination;

impl DeadBlockElimination {
    /// Create a new dead block elimination pass.
    pub fn new() -> Self {
        Self
    }

    /// Find all reachable blocks starting from the entry.
    fn find_reachable_blocks(&self, module: &IrModule) -> HashSet<BlockId> {
        let mut reachable = HashSet::new();
        let mut worklist = vec![module.entry_block];

        while let Some(block_id) = worklist.pop() {
            if !reachable.insert(block_id) {
                continue;
            }

            if let Some(block) = module.get_block(block_id) {
                // Add successors to worklist
                match &block.terminator {
                    Some(Terminator::Branch(target)) => {
                        worklist.push(*target);
                    }
                    Some(Terminator::CondBranch(_, then_target, else_target)) => {
                        worklist.push(*then_target);
                        worklist.push(*else_target);
                    }
                    Some(Terminator::Switch(_, default, cases)) => {
                        worklist.push(*default);
                        for (_, target) in cases {
                            worklist.push(*target);
                        }
                    }
                    _ => {}
                }
            }
        }

        reachable
    }
}

impl Default for DeadBlockElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for DeadBlockElimination {
    fn run(&self, module: &mut IrModule) -> OptimizationResult {
        let reachable = self.find_reachable_blocks(module);
        let mut result = OptimizationResult::unchanged();

        // Collect unreachable blocks
        let unreachable: Vec<BlockId> = module
            .blocks
            .keys()
            .filter(|id| !reachable.contains(id))
            .copied()
            .collect();

        // Remove unreachable blocks
        for block_id in unreachable {
            module.blocks.remove(&block_id);
            result.changed = true;
            result.blocks_removed += 1;
        }

        result
    }

    fn name(&self) -> &'static str {
        "dead-block-elimination"
    }
}

// ============================================================================
// ALGEBRAIC SIMPLIFICATION
// ============================================================================

/// Algebraic Simplification pass.
///
/// Simplifies expressions using algebraic identities:
/// - x + 0 = x
/// - x * 1 = x
/// - x * 0 = 0
/// - x - x = 0
/// - x / 1 = x
/// - x & 0 = 0
/// - x | 0 = x
/// - etc.
pub struct AlgebraicSimplification;

impl AlgebraicSimplification {
    /// Create a new algebraic simplification pass.
    pub fn new() -> Self {
        Self
    }

    /// Check if a constant is zero.
    fn is_zero(c: &ConstantValue) -> bool {
        match c {
            ConstantValue::I32(0) => true,
            ConstantValue::U32(0) => true,
            ConstantValue::I64(0) => true,
            ConstantValue::U64(0) => true,
            ConstantValue::F32(f) => *f == 0.0,
            ConstantValue::F64(f) => *f == 0.0,
            _ => false,
        }
    }

    /// Check if a constant is one.
    fn is_one(c: &ConstantValue) -> bool {
        match c {
            ConstantValue::I32(1) => true,
            ConstantValue::U32(1) => true,
            ConstantValue::I64(1) => true,
            ConstantValue::U64(1) => true,
            ConstantValue::F32(f) => *f == 1.0,
            ConstantValue::F64(f) => *f == 1.0,
            _ => false,
        }
    }

    /// Create a zero constant of the given type.
    fn zero_for_type(ty: &IrType) -> Option<ConstantValue> {
        Some(match ty {
            IrType::Scalar(ScalarType::I32) => ConstantValue::I32(0),
            IrType::Scalar(ScalarType::U32) => ConstantValue::U32(0),
            IrType::Scalar(ScalarType::I64) => ConstantValue::I64(0),
            IrType::Scalar(ScalarType::U64) => ConstantValue::U64(0),
            IrType::Scalar(ScalarType::F32) => ConstantValue::F32(0.0),
            IrType::Scalar(ScalarType::F64) => ConstantValue::F64(0.0),
            _ => return None,
        })
    }
}

impl Default for AlgebraicSimplification {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for AlgebraicSimplification {
    fn run(&self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        // Collect constants
        let mut constants = HashMap::new();
        for value in module.values.values() {
            if let IrNode::Constant(ref c) = value.node {
                constants.insert(value.id, c.clone());
            }
        }

        // Simplify operations
        for block in module.blocks.values_mut() {
            for inst in &mut block.instructions {
                let simplified = match &inst.node {
                    IrNode::BinaryOp(op, lhs, rhs) => {
                        let lhs_const = constants.get(lhs);
                        let rhs_const = constants.get(rhs);

                        match op {
                            // x + 0 = x
                            BinaryOp::Add if rhs_const.is_some_and(Self::is_zero) => {
                                Some(IrNode::Parameter(0)) // Placeholder, replaced below
                            }
                            // 0 + x = x
                            BinaryOp::Add if lhs_const.is_some_and(Self::is_zero) => {
                                Some(IrNode::Parameter(1))
                            }
                            // x * 1 = x
                            BinaryOp::Mul if rhs_const.is_some_and(Self::is_one) => {
                                Some(IrNode::Parameter(0))
                            }
                            // 1 * x = x
                            BinaryOp::Mul if lhs_const.is_some_and(Self::is_one) => {
                                Some(IrNode::Parameter(1))
                            }
                            // x * 0 = 0
                            BinaryOp::Mul
                                if rhs_const.is_some_and(Self::is_zero)
                                    || lhs_const.is_some_and(Self::is_zero) =>
                            {
                                Self::zero_for_type(&inst.result_type).map(IrNode::Constant)
                            }
                            // x - 0 = x
                            BinaryOp::Sub if rhs_const.is_some_and(Self::is_zero) => {
                                Some(IrNode::Parameter(0))
                            }
                            // x / 1 = x
                            BinaryOp::Div if rhs_const.is_some_and(Self::is_one) => {
                                Some(IrNode::Parameter(0))
                            }
                            // x & 0 = 0
                            BinaryOp::And if rhs_const.is_some_and(Self::is_zero) => {
                                Self::zero_for_type(&inst.result_type).map(IrNode::Constant)
                            }
                            // x | 0 = x
                            BinaryOp::Or if rhs_const.is_some_and(Self::is_zero) => {
                                Some(IrNode::Parameter(0))
                            }
                            // x ^ 0 = x
                            BinaryOp::Xor if rhs_const.is_some_and(Self::is_zero) => {
                                Some(IrNode::Parameter(0))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };

                // Apply simplification
                if let Some(simplified_node) = simplified {
                    match simplified_node {
                        IrNode::Parameter(0) => {
                            // Replace with lhs
                            // Note: Full value propagation requires SSA-form copy propagation
                            // which is a more complex optimization. For now, we only handle
                            // constant folding cases. The instruction remains unchanged.
                        }
                        IrNode::Parameter(1) => {
                            // Replace with rhs
                            // Same limitation as above - would need copy propagation pass
                        }
                        IrNode::Constant(c) => {
                            inst.node = IrNode::Constant(c.clone());
                            constants.insert(inst.result, c);
                            result.changed = true;
                            result.instructions_modified += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        result
    }

    fn name(&self) -> &'static str {
        "algebraic-simplification"
    }
}

// ============================================================================
// PASS MANAGER
// ============================================================================

/// Runs optimization passes on an IR module.
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl PassManager {
    /// Create a new pass manager with default passes.
    pub fn new() -> Self {
        Self {
            passes: vec![
                Box::new(ConstantFolding::new()),
                Box::new(AlgebraicSimplification::new()),
                Box::new(DeadCodeElimination::new()),
                Box::new(DeadBlockElimination::new()),
            ],
            max_iterations: 10,
        }
    }

    /// Create an empty pass manager.
    pub fn empty() -> Self {
        Self {
            passes: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Add a pass to the manager.
    pub fn add_pass<P: OptimizationPass + 'static>(&mut self, pass: P) -> &mut Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iterations(&mut self, n: usize) -> &mut Self {
        self.max_iterations = n;
        self
    }

    /// Run all passes on the module.
    pub fn run(&self, module: &mut IrModule) -> OptimizationResult {
        let mut total_result = OptimizationResult::unchanged();

        for iteration in 0..self.max_iterations {
            let mut changed = false;

            for pass in &self.passes {
                let pass_result = pass.run(module);
                changed |= pass_result.changed;
                total_result.merge(pass_result);
            }

            if !changed {
                break;
            }

            // Safety check
            if iteration == self.max_iterations - 1 {
                tracing::warn!(
                    max_iterations = self.max_iterations,
                    "optimization reached max iterations"
                );
            }
        }

        total_result
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Run all standard optimization passes on a module.
pub fn optimize(module: &mut IrModule) -> OptimizationResult {
    PassManager::new().run(module)
}

/// Run only DCE on a module.
pub fn run_dce(module: &mut IrModule) -> OptimizationResult {
    DeadCodeElimination::new().run(module)
}

/// Run only constant folding on a module.
pub fn run_constant_folding(module: &mut IrModule) -> OptimizationResult {
    ConstantFolding::new().run(module)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrBuilder;

    #[test]
    fn test_dce_removes_unused() {
        let mut builder = IrBuilder::new("test");

        // Create some values (constants are stored in values map, not as instructions)
        let a = builder.const_i32(10);
        let b = builder.const_i32(20);

        // Create an unused computation - this adds an instruction to the block
        let _unused_sum = builder.add(a, b);

        // Create a used computation
        let c = builder.const_i32(5);
        let used = builder.mul(c, c);

        // Return the used value
        builder.ret_value(used);

        let mut module = builder.build();

        let result = DeadCodeElimination::new().run(&mut module);

        // The unused add instruction should be removed
        assert!(result.changed);
        assert!(result.instructions_removed > 0);
    }

    #[test]
    fn test_constant_folding_binary() {
        let mut builder = IrBuilder::new("test");

        // 2 + 3 should fold to 5
        let a = builder.const_i32(2);
        let b = builder.const_i32(3);
        let sum = builder.add(a, b);

        builder.ret_value(sum);

        let mut module = builder.build();

        let result = ConstantFolding::new().run(&mut module);

        assert!(result.changed);
        assert!(result.instructions_modified > 0);
    }

    #[test]
    fn test_constant_folding_unary() {
        let mut builder = IrBuilder::new("test");

        // -5 should fold
        let a = builder.const_i32(5);
        let neg = builder.neg(a);

        builder.ret_value(neg);

        let mut module = builder.build();

        let result = ConstantFolding::new().run(&mut module);

        assert!(result.changed);
    }

    #[test]
    fn test_pass_manager() {
        let mut builder = IrBuilder::new("test");

        // Create some optimizable code
        let a = builder.const_i32(2);
        let b = builder.const_i32(3);
        let sum = builder.add(a, b);
        let _unused = builder.const_i32(999);

        builder.ret_value(sum);

        let mut module = builder.build();

        let result = PassManager::new().run(&mut module);

        assert!(result.changed);
    }

    #[test]
    fn test_optimization_result_merge() {
        let mut r1 = OptimizationResult {
            changed: true,
            instructions_removed: 5,
            instructions_modified: 3,
            blocks_removed: 1,
        };

        let r2 = OptimizationResult {
            changed: false,
            instructions_removed: 2,
            instructions_modified: 1,
            blocks_removed: 0,
        };

        r1.merge(r2);

        assert!(r1.changed);
        assert_eq!(r1.instructions_removed, 7);
        assert_eq!(r1.instructions_modified, 4);
        assert_eq!(r1.blocks_removed, 1);
    }
}
