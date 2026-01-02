//! RingKernel Intermediate Representation (IR)
//!
//! This crate provides a unified IR for GPU code generation across multiple backends
//! (CUDA, WGSL, MSL). The IR is SSA-based and captures GPU-specific operations.
//!
//! # Architecture
//!
//! ```text
//! Rust DSL → IR → Backend-specific lowering → CUDA/WGSL/MSL
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ir::{IrBuilder, IrType, Dimension};
//!
//! let mut builder = IrBuilder::new("saxpy");
//!
//! // Define parameters
//! let x = builder.parameter("x", IrType::Ptr(Box::new(IrType::F32)));
//! let y = builder.parameter("y", IrType::Ptr(Box::new(IrType::F32)));
//! let a = builder.parameter("a", IrType::F32);
//! let n = builder.parameter("n", IrType::I32);
//!
//! // Get thread index
//! let idx = builder.thread_id(Dimension::X);
//!
//! // Bounds check
//! let in_bounds = builder.lt(idx, n);
//! builder.if_then(in_bounds, |b| {
//!     let x_val = b.load(x, idx);
//!     let y_val = b.load(y, idx);
//!     let result = b.add(b.mul(a, x_val), y_val);
//!     b.store(y, idx, result);
//! });
//!
//! let ir = builder.build();
//! ```

#![warn(missing_docs)]

mod builder;
mod capabilities;
mod error;
mod nodes;
mod printer;
mod types;
mod validation;

pub use builder::{IrBuilder, IrBuilderScope};
pub use capabilities::{BackendCapabilities, Capabilities, CapabilityFlag};
pub use error::{IrError, IrResult};
pub use nodes::*;
pub use printer::IrPrinter;
pub use types::{IrType, ScalarType, VectorType};
pub use validation::{ValidationLevel, ValidationResult, Validator};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for IR values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(u64);

impl ValueId {
    /// Create a new unique value ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for ValueId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Unique identifier for IR blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u64);

impl BlockId {
    /// Create a new unique block ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for BlockId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// A complete IR module representing a GPU kernel.
#[derive(Debug, Clone)]
pub struct IrModule {
    /// Module name (kernel name).
    pub name: String,
    /// Function parameters.
    pub parameters: Vec<Parameter>,
    /// Entry block.
    pub entry_block: BlockId,
    /// All blocks in the module.
    pub blocks: HashMap<BlockId, Block>,
    /// All values defined in the module.
    pub values: HashMap<ValueId, Value>,
    /// Required capabilities for this module.
    pub required_capabilities: Capabilities,
    /// Kernel configuration.
    pub config: KernelConfig,
}

impl IrModule {
    /// Create a new empty module.
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId::new();
        let mut blocks = HashMap::new();
        blocks.insert(entry, Block::new(entry, "entry"));

        Self {
            name: name.into(),
            parameters: Vec::new(),
            entry_block: entry,
            blocks,
            values: HashMap::new(),
            required_capabilities: Capabilities::default(),
            config: KernelConfig::default(),
        }
    }

    /// Get a block by ID.
    pub fn get_block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID.
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        self.blocks.get_mut(&id)
    }

    /// Get a value by ID.
    pub fn get_value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(&id)
    }

    /// Add a value to the module.
    pub fn add_value(&mut self, value: Value) -> ValueId {
        let id = value.id;
        self.values.insert(id, value);
        id
    }

    /// Get the entry block.
    pub fn entry(&self) -> &Block {
        self.blocks.get(&self.entry_block).expect("entry block must exist")
    }

    /// Validate the module.
    pub fn validate(&self, level: ValidationLevel) -> ValidationResult {
        Validator::new(level).validate(self)
    }

    /// Pretty-print the IR.
    pub fn pretty_print(&self) -> String {
        IrPrinter::new().print(self)
    }
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: IrType,
    /// Value ID for this parameter.
    pub value_id: ValueId,
    /// Parameter index.
    pub index: usize,
}

/// A basic block containing IR nodes.
#[derive(Debug, Clone)]
pub struct Block {
    /// Block identifier.
    pub id: BlockId,
    /// Block label (for debugging).
    pub label: String,
    /// Instructions in this block.
    pub instructions: Vec<Instruction>,
    /// Terminator instruction.
    pub terminator: Option<Terminator>,
    /// Predecessor blocks.
    pub predecessors: Vec<BlockId>,
    /// Successor blocks.
    pub successors: Vec<BlockId>,
}

impl Block {
    /// Create a new block.
    pub fn new(id: BlockId, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            instructions: Vec::new(),
            terminator: None,
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Add an instruction to the block.
    pub fn add_instruction(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Set the terminator.
    pub fn set_terminator(&mut self, term: Terminator) {
        self.terminator = Some(term);
    }

    /// Check if block is terminated.
    pub fn is_terminated(&self) -> bool {
        self.terminator.is_some()
    }
}

/// An IR value with type information.
#[derive(Debug, Clone)]
pub struct Value {
    /// Value identifier.
    pub id: ValueId,
    /// Value type.
    pub ty: IrType,
    /// The node that produces this value.
    pub node: IrNode,
}

impl Value {
    /// Create a new value.
    pub fn new(ty: IrType, node: IrNode) -> Self {
        Self {
            id: ValueId::new(),
            ty,
            node,
        }
    }
}

/// Kernel configuration.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block size (threads per block).
    pub block_size: (u32, u32, u32),
    /// Grid size (blocks per grid), if static.
    pub grid_size: Option<(u32, u32, u32)>,
    /// Shared memory size in bytes.
    pub shared_memory_bytes: u32,
    /// Whether this is a persistent kernel.
    pub is_persistent: bool,
    /// Kernel mode.
    pub mode: KernelMode,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: (256, 1, 1),
            grid_size: None,
            shared_memory_bytes: 0,
            is_persistent: false,
            mode: KernelMode::Compute,
        }
    }
}

/// Kernel execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelMode {
    /// Standard compute kernel.
    Compute,
    /// Persistent message-processing kernel.
    Persistent,
    /// Stencil computation kernel.
    Stencil,
}

/// Dimension for GPU indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    /// X dimension.
    X,
    /// Y dimension.
    Y,
    /// Z dimension.
    Z,
}

impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dimension::X => write!(f, "x"),
            Dimension::Y => write!(f, "y"),
            Dimension::Z => write!(f, "z"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_id_unique() {
        let id1 = ValueId::new();
        let id2 = ValueId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_block_id_unique() {
        let id1 = BlockId::new();
        let id2 = BlockId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_ir_module_new() {
        let module = IrModule::new("test_kernel");
        assert_eq!(module.name, "test_kernel");
        assert!(module.parameters.is_empty());
        assert!(module.blocks.contains_key(&module.entry_block));
    }

    #[test]
    fn test_block_operations() {
        let id = BlockId::new();
        let mut block = Block::new(id, "test");

        assert!(!block.is_terminated());
        assert!(block.instructions.is_empty());

        block.set_terminator(Terminator::Return(None));
        assert!(block.is_terminated());
    }

    #[test]
    fn test_dimension_display() {
        assert_eq!(format!("{}", Dimension::X), "x");
        assert_eq!(format!("{}", Dimension::Y), "y");
        assert_eq!(format!("{}", Dimension::Z), "z");
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.block_size, (256, 1, 1));
        assert!(config.grid_size.is_none());
        assert_eq!(config.shared_memory_bytes, 0);
        assert!(!config.is_persistent);
    }
}
