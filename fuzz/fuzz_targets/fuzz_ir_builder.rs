//! Fuzz target for IR builder operations.
//!
//! Tests random sequences of IR building operations to find crashes
//! or panics in the IR construction logic.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ringkernel_ir::{Dimension, IrBuilder, IrType, ScalarType};

/// Operations that can be performed on the IR builder.
#[derive(Debug, Arbitrary)]
enum IrBuilderOp {
    // Parameters
    AddParameter { name_idx: u8, ty: FuzzIrType },

    // Constants
    ConstI32(i32),
    ConstU32(u32),
    ConstF32(f32),
    ConstBool(bool),

    // Binary ops (use indices into value pool)
    Add { lhs: u8, rhs: u8 },
    Sub { lhs: u8, rhs: u8 },
    Mul { lhs: u8, rhs: u8 },
    Div { lhs: u8, rhs: u8 },
    And { lhs: u8, rhs: u8 },
    Or { lhs: u8, rhs: u8 },
    Xor { lhs: u8, rhs: u8 },

    // Unary ops
    Neg { operand: u8 },

    // Comparisons
    Lt { lhs: u8, rhs: u8 },
    Le { lhs: u8, rhs: u8 },
    Gt { lhs: u8, rhs: u8 },
    Ge { lhs: u8, rhs: u8 },
    Eq { lhs: u8, rhs: u8 },
    Ne { lhs: u8, rhs: u8 },

    // GPU intrinsics
    ThreadId(FuzzDimension),
    BlockId(FuzzDimension),
    BlockDim(FuzzDimension),
    GridDim(FuzzDimension),

    // Control flow
    CreateBlock { name_idx: u8 },
    SwitchToBlock { block_idx: u8 },
    Branch { target_idx: u8 },

    // Terminator
    Return,
    ReturnValue { value: u8 },
}

#[derive(Debug, Arbitrary)]
enum FuzzIrType {
    I32,
    U32,
    F32,
    Bool,
    PtrI32,
    PtrF32,
}

impl FuzzIrType {
    fn to_ir_type(&self) -> IrType {
        match self {
            FuzzIrType::I32 => IrType::I32,
            FuzzIrType::U32 => IrType::U32,
            FuzzIrType::F32 => IrType::F32,
            FuzzIrType::Bool => IrType::BOOL,
            FuzzIrType::PtrI32 => IrType::ptr(IrType::I32),
            FuzzIrType::PtrF32 => IrType::ptr(IrType::F32),
        }
    }
}

#[derive(Debug, Arbitrary)]
enum FuzzDimension {
    X,
    Y,
    Z,
}

impl FuzzDimension {
    fn to_dimension(&self) -> Dimension {
        match self {
            FuzzDimension::X => Dimension::X,
            FuzzDimension::Y => Dimension::Y,
            FuzzDimension::Z => Dimension::Z,
        }
    }
}

/// Fuzz input: a sequence of operations to apply to the builder.
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    kernel_name_len: u8,
    ops: Vec<IrBuilderOp>,
}

const PARAM_NAMES: &[&str] = &["a", "b", "c", "d", "x", "y", "z", "n", "data", "out"];
const BLOCK_NAMES: &[&str] = &["loop", "then", "else", "cont", "exit", "body"];

fuzz_target!(|input: FuzzInput| {
    // Limit operations to prevent timeout
    if input.ops.len() > 100 {
        return;
    }

    // Create builder with sanitized name
    let name_len = (input.kernel_name_len % 16) as usize + 1;
    let name = format!("kernel_{:0>width$}", 0, width = name_len);
    let mut builder = IrBuilder::new(&name);

    // Track created values and blocks
    let mut values = Vec::new();
    let mut blocks = vec![builder.current_block()];

    for op in &input.ops {
        // Helper to get a valid value index
        let get_value = |idx: u8| -> Option<ringkernel_ir::ValueId> {
            if values.is_empty() {
                None
            } else {
                Some(values[idx as usize % values.len()])
            }
        };

        // Helper to get a valid block index
        let get_block = |idx: u8| -> ringkernel_ir::BlockId {
            blocks[idx as usize % blocks.len()]
        };

        match op {
            IrBuilderOp::AddParameter { name_idx, ty } => {
                let name = PARAM_NAMES[*name_idx as usize % PARAM_NAMES.len()];
                let id = builder.parameter(name, ty.to_ir_type());
                values.push(id);
            }

            IrBuilderOp::ConstI32(v) => {
                let id = builder.const_i32(*v);
                values.push(id);
            }
            IrBuilderOp::ConstU32(v) => {
                let id = builder.const_u32(*v);
                values.push(id);
            }
            IrBuilderOp::ConstF32(v) => {
                // Avoid NaN/Inf which might cause issues
                let v = if v.is_nan() || v.is_infinite() { 0.0 } else { *v };
                let id = builder.const_f32(v);
                values.push(id);
            }
            IrBuilderOp::ConstBool(v) => {
                let id = builder.const_bool(*v);
                values.push(id);
            }

            IrBuilderOp::Add { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.add(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Sub { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.sub(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Mul { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.mul(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Div { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.div(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::And { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.and(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Or { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.or(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Xor { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.xor(l, r);
                    values.push(id);
                }
            }

            IrBuilderOp::Neg { operand } => {
                if let Some(op) = get_value(*operand) {
                    let id = builder.neg(op);
                    values.push(id);
                }
            }

            IrBuilderOp::Lt { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.lt(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Le { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.le(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Gt { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.gt(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Ge { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.ge(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Eq { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.eq(l, r);
                    values.push(id);
                }
            }
            IrBuilderOp::Ne { lhs, rhs } => {
                if let (Some(l), Some(r)) = (get_value(*lhs), get_value(*rhs)) {
                    let id = builder.ne(l, r);
                    values.push(id);
                }
            }

            IrBuilderOp::ThreadId(dim) => {
                let id = builder.thread_id(dim.to_dimension());
                values.push(id);
            }
            IrBuilderOp::BlockId(dim) => {
                let id = builder.block_id(dim.to_dimension());
                values.push(id);
            }
            IrBuilderOp::BlockDim(dim) => {
                let id = builder.block_dim(dim.to_dimension());
                values.push(id);
            }
            IrBuilderOp::GridDim(dim) => {
                let id = builder.grid_dim(dim.to_dimension());
                values.push(id);
            }

            IrBuilderOp::CreateBlock { name_idx } => {
                let name = BLOCK_NAMES[*name_idx as usize % BLOCK_NAMES.len()];
                let id = builder.create_block(name);
                blocks.push(id);
            }
            IrBuilderOp::SwitchToBlock { block_idx } => {
                let block = get_block(*block_idx);
                builder.switch_to_block(block);
            }
            IrBuilderOp::Branch { target_idx } => {
                let target = get_block(*target_idx);
                builder.branch(target);
            }

            IrBuilderOp::Return => {
                builder.ret();
            }
            IrBuilderOp::ReturnValue { value } => {
                if let Some(v) = get_value(*value) {
                    builder.ret_value(v);
                }
            }
        }
    }

    // Build the module - should not panic
    let module = builder.build();

    // Try to pretty print - should not panic
    let _ = module.pretty_print();

    // Try validation - should not panic (may return errors)
    let _ = module.validate(ringkernel_ir::ValidationLevel::Full);
});
