//! IR builder API.
//!
//! Provides an ergonomic interface for constructing IR modules.

use crate::{
    nodes::*, Block, BlockId, CapabilityFlag, Dimension, Instruction, IrModule, IrType,
    KernelConfig, KernelMode, Parameter, Terminator, Value, ValueId,
};

/// Builder for constructing IR modules.
pub struct IrBuilder {
    module: IrModule,
    current_block: BlockId,
}

impl IrBuilder {
    /// Create a new builder.
    pub fn new(name: impl Into<String>) -> Self {
        let module = IrModule::new(name);
        let entry = module.entry_block;
        Self {
            module,
            current_block: entry,
        }
    }

    /// Build and return the IR module.
    pub fn build(self) -> IrModule {
        self.module
    }

    /// Get a reference to the module being built.
    pub fn module(&self) -> &IrModule {
        &self.module
    }

    /// Set kernel configuration.
    pub fn set_config(&mut self, config: KernelConfig) {
        self.module.config = config;
    }

    /// Set block size.
    pub fn set_block_size(&mut self, x: u32, y: u32, z: u32) {
        self.module.config.block_size = (x, y, z);
    }

    /// Mark as persistent kernel.
    pub fn set_persistent(&mut self, persistent: bool) {
        self.module.config.is_persistent = persistent;
        if persistent {
            self.module.config.mode = KernelMode::Persistent;
        }
    }

    // ========================================================================
    // Parameters
    // ========================================================================

    /// Add a parameter.
    pub fn parameter(&mut self, name: impl Into<String>, ty: IrType) -> ValueId {
        let value_id = ValueId::new();
        let index = self.module.parameters.len();

        self.module.parameters.push(Parameter {
            name: name.into(),
            ty: ty.clone(),
            value_id,
            index,
        });

        let value = Value::new(ty, IrNode::Parameter(index));
        self.module.values.insert(value_id, value);

        value_id
    }

    // ========================================================================
    // Blocks
    // ========================================================================

    /// Create a new block.
    pub fn create_block(&mut self, label: impl Into<String>) -> BlockId {
        let id = BlockId::new();
        self.module.blocks.insert(id, Block::new(id, label));
        id
    }

    /// Switch to a different block.
    pub fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    /// Get current block ID.
    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    // ========================================================================
    // Constants
    // ========================================================================

    /// Create an i32 constant.
    pub fn const_i32(&mut self, value: i32) -> ValueId {
        self.add_value(IrType::I32, IrNode::Constant(ConstantValue::I32(value)))
    }

    /// Create an i64 constant.
    pub fn const_i64(&mut self, value: i64) -> ValueId {
        self.module.required_capabilities.add(CapabilityFlag::Int64);
        self.add_value(IrType::I64, IrNode::Constant(ConstantValue::I64(value)))
    }

    /// Create a u32 constant.
    pub fn const_u32(&mut self, value: u32) -> ValueId {
        self.add_value(IrType::U32, IrNode::Constant(ConstantValue::U32(value)))
    }

    /// Create a u64 constant.
    pub fn const_u64(&mut self, value: u64) -> ValueId {
        self.module.required_capabilities.add(CapabilityFlag::Int64);
        self.add_value(IrType::U64, IrNode::Constant(ConstantValue::U64(value)))
    }

    /// Create an f32 constant.
    pub fn const_f32(&mut self, value: f32) -> ValueId {
        self.add_value(IrType::F32, IrNode::Constant(ConstantValue::F32(value)))
    }

    /// Create an f64 constant.
    pub fn const_f64(&mut self, value: f64) -> ValueId {
        self.module
            .required_capabilities
            .add(CapabilityFlag::Float64);
        self.add_value(IrType::F64, IrNode::Constant(ConstantValue::F64(value)))
    }

    /// Create a boolean constant.
    pub fn const_bool(&mut self, value: bool) -> ValueId {
        self.add_value(IrType::BOOL, IrNode::Constant(ConstantValue::Bool(value)))
    }

    // ========================================================================
    // Binary Operations
    // ========================================================================

    /// Add two values.
    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Add, lhs, rhs))
    }

    /// Subtract two values.
    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Sub, lhs, rhs))
    }

    /// Multiply two values.
    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Mul, lhs, rhs))
    }

    /// Divide two values.
    pub fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Div, lhs, rhs))
    }

    /// Remainder.
    pub fn rem(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Rem, lhs, rhs))
    }

    /// Bitwise AND.
    pub fn and(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::And, lhs, rhs))
    }

    /// Bitwise OR.
    pub fn or(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Or, lhs, rhs))
    }

    /// Bitwise XOR.
    pub fn xor(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Xor, lhs, rhs))
    }

    /// Left shift.
    pub fn shl(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Shl, lhs, rhs))
    }

    /// Logical right shift.
    pub fn shr(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Shr, lhs, rhs))
    }

    /// Minimum.
    pub fn min(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Min, lhs, rhs))
    }

    /// Maximum.
    pub fn max(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let ty = self.get_value_type(lhs);
        self.add_instruction(ty, IrNode::BinaryOp(BinaryOp::Max, lhs, rhs))
    }

    // ========================================================================
    // Unary Operations
    // ========================================================================

    /// Negate.
    pub fn neg(&mut self, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::UnaryOp(UnaryOp::Neg, value))
    }

    /// Bitwise NOT.
    pub fn not(&mut self, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::UnaryOp(UnaryOp::Not, value))
    }

    /// Absolute value.
    pub fn abs(&mut self, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::UnaryOp(UnaryOp::Abs, value))
    }

    /// Square root.
    pub fn sqrt(&mut self, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::UnaryOp(UnaryOp::Sqrt, value))
    }

    /// Floor.
    pub fn floor(&mut self, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::UnaryOp(UnaryOp::Floor, value))
    }

    /// Ceiling.
    pub fn ceil(&mut self, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::UnaryOp(UnaryOp::Ceil, value))
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    /// Equal comparison.
    pub fn eq(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::Compare(CompareOp::Eq, lhs, rhs))
    }

    /// Not equal comparison.
    pub fn ne(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::Compare(CompareOp::Ne, lhs, rhs))
    }

    /// Less than.
    pub fn lt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::Compare(CompareOp::Lt, lhs, rhs))
    }

    /// Less than or equal.
    pub fn le(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::Compare(CompareOp::Le, lhs, rhs))
    }

    /// Greater than.
    pub fn gt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::Compare(CompareOp::Gt, lhs, rhs))
    }

    /// Greater than or equal.
    pub fn ge(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::Compare(CompareOp::Ge, lhs, rhs))
    }

    // ========================================================================
    // Memory
    // ========================================================================

    /// Load from pointer.
    pub fn load(&mut self, ptr: ValueId) -> ValueId {
        let ptr_ty = self.get_value_type(ptr);
        let elem_ty = match ptr_ty {
            IrType::Ptr(inner) => (*inner).clone(),
            _ => IrType::Void,
        };
        self.add_instruction(elem_ty, IrNode::Load(ptr))
    }

    /// Store to pointer.
    pub fn store(&mut self, ptr: ValueId, value: ValueId) {
        self.add_instruction(IrType::Void, IrNode::Store(ptr, value));
    }

    /// Get element pointer.
    pub fn gep(&mut self, ptr: ValueId, indices: Vec<ValueId>) -> ValueId {
        let ty = self.get_value_type(ptr);
        self.add_instruction(ty, IrNode::GetElementPtr(ptr, indices))
    }

    /// Allocate shared memory.
    pub fn shared_alloc(&mut self, ty: IrType, count: usize) -> ValueId {
        self.module
            .required_capabilities
            .add(CapabilityFlag::SharedMemory);
        let ptr_ty = IrType::ptr(ty.clone());
        self.add_instruction(ptr_ty, IrNode::SharedAlloc(ty, count))
    }

    // ========================================================================
    // GPU Indexing
    // ========================================================================

    /// Get thread ID.
    pub fn thread_id(&mut self, dim: Dimension) -> ValueId {
        self.add_instruction(IrType::U32, IrNode::ThreadId(dim))
    }

    /// Get block ID.
    pub fn block_id(&mut self, dim: Dimension) -> ValueId {
        self.add_instruction(IrType::U32, IrNode::BlockId(dim))
    }

    /// Get block dimension.
    pub fn block_dim(&mut self, dim: Dimension) -> ValueId {
        self.add_instruction(IrType::U32, IrNode::BlockDim(dim))
    }

    /// Get grid dimension.
    pub fn grid_dim(&mut self, dim: Dimension) -> ValueId {
        self.add_instruction(IrType::U32, IrNode::GridDim(dim))
    }

    /// Get global thread ID.
    pub fn global_thread_id(&mut self, dim: Dimension) -> ValueId {
        self.add_instruction(IrType::U32, IrNode::GlobalThreadId(dim))
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Block/threadgroup barrier.
    pub fn barrier(&mut self) {
        self.add_instruction(IrType::Void, IrNode::Barrier);
    }

    /// Memory fence.
    pub fn fence(&mut self, scope: MemoryScope) {
        self.add_instruction(IrType::Void, IrNode::MemoryFence(scope));
    }

    /// Grid sync (cooperative groups).
    pub fn grid_sync(&mut self) {
        self.module
            .required_capabilities
            .add(CapabilityFlag::CooperativeGroups);
        self.add_instruction(IrType::Void, IrNode::GridSync);
    }

    // ========================================================================
    // Atomics
    // ========================================================================

    /// Atomic add.
    pub fn atomic_add(&mut self, ptr: ValueId, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::Atomic(AtomicOp::Add, ptr, value))
    }

    /// Atomic exchange.
    pub fn atomic_exchange(&mut self, ptr: ValueId, value: ValueId) -> ValueId {
        let ty = self.get_value_type(value);
        self.add_instruction(ty, IrNode::Atomic(AtomicOp::Exchange, ptr, value))
    }

    /// Atomic compare-and-swap.
    pub fn atomic_cas(&mut self, ptr: ValueId, expected: ValueId, desired: ValueId) -> ValueId {
        let ty = self.get_value_type(expected);
        self.add_instruction(ty, IrNode::AtomicCas(ptr, expected, desired))
    }

    // ========================================================================
    // Control Flow
    // ========================================================================

    /// Select (ternary).
    pub fn select(&mut self, cond: ValueId, then_val: ValueId, else_val: ValueId) -> ValueId {
        let ty = self.get_value_type(then_val);
        self.add_instruction(ty, IrNode::Select(cond, then_val, else_val))
    }

    /// Branch to block.
    pub fn branch(&mut self, target: BlockId) {
        self.set_terminator(Terminator::Branch(target));
        self.add_successor(target);
    }

    /// Conditional branch.
    pub fn cond_branch(&mut self, cond: ValueId, then_block: BlockId, else_block: BlockId) {
        self.set_terminator(Terminator::CondBranch(cond, then_block, else_block));
        self.add_successor(then_block);
        self.add_successor(else_block);
    }

    /// Return from kernel.
    pub fn ret(&mut self) {
        self.set_terminator(Terminator::Return(None));
    }

    /// Return value from kernel.
    pub fn ret_value(&mut self, value: ValueId) {
        self.set_terminator(Terminator::Return(Some(value)));
    }

    // ========================================================================
    // RingKernel Messaging
    // ========================================================================

    /// Enqueue to output (K2H).
    pub fn k2h_enqueue(&mut self, message: ValueId) {
        self.add_instruction(IrType::Void, IrNode::K2HEnqueue(message));
    }

    /// Dequeue from input (H2K).
    pub fn h2k_dequeue(&mut self, msg_ty: IrType) -> ValueId {
        self.add_instruction(msg_ty, IrNode::H2KDequeue)
    }

    /// Check if input queue is empty.
    pub fn h2k_is_empty(&mut self) -> ValueId {
        self.add_instruction(IrType::BOOL, IrNode::H2KIsEmpty)
    }

    /// Send K2K message.
    pub fn k2k_send(&mut self, dest: ValueId, message: ValueId) {
        self.add_instruction(IrType::Void, IrNode::K2KSend(dest, message));
    }

    /// Try receive K2K message.
    pub fn k2k_try_recv(&mut self, msg_ty: IrType) -> ValueId {
        self.add_instruction(msg_ty, IrNode::K2KTryRecv)
    }

    // ========================================================================
    // HLC Operations
    // ========================================================================

    /// Get current HLC time.
    pub fn hlc_now(&mut self) -> ValueId {
        self.add_instruction(IrType::U64, IrNode::HlcNow)
    }

    /// Tick HLC.
    pub fn hlc_tick(&mut self) -> ValueId {
        self.add_instruction(IrType::U64, IrNode::HlcTick)
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    fn add_value(&mut self, ty: IrType, node: IrNode) -> ValueId {
        let value = Value::new(ty, node);
        let id = value.id;
        self.module.values.insert(id, value);
        id
    }

    fn add_instruction(&mut self, ty: IrType, node: IrNode) -> ValueId {
        let result = ValueId::new();
        let inst = Instruction::new(result, ty.clone(), node.clone());

        if let Some(block) = self.module.blocks.get_mut(&self.current_block) {
            block.add_instruction(inst);
        }

        // Also add to values map
        let value = Value::new(ty, node);
        self.module.values.insert(result, value);

        result
    }

    fn set_terminator(&mut self, term: Terminator) {
        if let Some(block) = self.module.blocks.get_mut(&self.current_block) {
            block.set_terminator(term);
        }
    }

    fn add_successor(&mut self, succ: BlockId) {
        let current = self.current_block;
        if let Some(block) = self.module.blocks.get_mut(&current) {
            block.successors.push(succ);
        }
        if let Some(succ_block) = self.module.blocks.get_mut(&succ) {
            succ_block.predecessors.push(current);
        }
    }

    fn get_value_type(&self, id: ValueId) -> IrType {
        self.module
            .values
            .get(&id)
            .map(|v| v.ty.clone())
            .unwrap_or(IrType::Void)
    }
}

/// Scoped builder for structured control flow.
pub struct IrBuilderScope<'a> {
    builder: &'a mut IrBuilder,
}

impl<'a> IrBuilderScope<'a> {
    /// Create a new scope.
    pub fn new(builder: &'a mut IrBuilder) -> Self {
        Self { builder }
    }

    /// Add two values.
    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.add(lhs, rhs)
    }

    /// Multiply two values.
    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.builder.mul(lhs, rhs)
    }

    /// Load from pointer.
    pub fn load(&mut self, ptr: ValueId) -> ValueId {
        self.builder.load(ptr)
    }

    /// Store to pointer.
    pub fn store(&mut self, ptr: ValueId, value: ValueId) {
        self.builder.store(ptr, value);
    }

    /// Get thread ID.
    pub fn thread_id(&mut self, dim: Dimension) -> ValueId {
        self.builder.thread_id(dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let mut builder = IrBuilder::new("test");

        let x = builder.parameter("x", IrType::ptr(IrType::F32));
        let y = builder.parameter("y", IrType::ptr(IrType::F32));

        let idx = builder.thread_id(Dimension::X);
        let x_val = builder.load(x);
        let y_val = builder.load(y);
        let result = builder.add(x_val, y_val);
        builder.store(y, result);
        builder.ret();

        let module = builder.build();
        assert_eq!(module.name, "test");
        assert_eq!(module.parameters.len(), 2);
    }

    #[test]
    fn test_builder_constants() {
        let mut builder = IrBuilder::new("test");

        let a = builder.const_i32(42);
        let b = builder.const_f32(3.14);
        let c = builder.const_bool(true);

        let module = builder.build();
        assert!(module.values.contains_key(&a));
        assert!(module.values.contains_key(&b));
        assert!(module.values.contains_key(&c));
    }

    #[test]
    fn test_builder_control_flow() {
        let mut builder = IrBuilder::new("test");

        let n = builder.parameter("n", IrType::I32);
        let idx = builder.thread_id(Dimension::X);
        let cond = builder.lt(idx, n);

        let then_block = builder.create_block("then");
        let end_block = builder.create_block("end");

        builder.cond_branch(cond, then_block, end_block);

        builder.switch_to_block(then_block);
        builder.branch(end_block);

        builder.switch_to_block(end_block);
        builder.ret();

        let module = builder.build();
        assert_eq!(module.blocks.len(), 3);
    }

    #[test]
    fn test_builder_capabilities() {
        let mut builder = IrBuilder::new("test");

        // f64 should add Float64 capability
        builder.const_f64(1.0);

        // grid_sync should add CooperativeGroups
        builder.grid_sync();

        let module = builder.build();
        assert!(module.required_capabilities.has(CapabilityFlag::Float64));
        assert!(module
            .required_capabilities
            .has(CapabilityFlag::CooperativeGroups));
    }

    #[test]
    fn test_builder_persistent_config() {
        let mut builder = IrBuilder::new("persistent_kernel");
        builder.set_persistent(true);
        builder.set_block_size(128, 1, 1);

        let module = builder.build();
        assert!(module.config.is_persistent);
        assert_eq!(module.config.mode, KernelMode::Persistent);
        assert_eq!(module.config.block_size, (128, 1, 1));
    }
}
