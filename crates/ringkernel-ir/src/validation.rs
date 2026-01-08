//! IR validation.
//!
//! Validates IR modules for correctness before lowering to backend code.

use std::collections::HashSet;

use crate::{Block, BlockId, IrModule, IrNode, IrType, Terminator, ValueId};

/// Validation strictness level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// No validation.
    None,
    /// Basic structural validation.
    Basic,
    /// Full type checking and SSA validation.
    Full,
    /// Strict validation with warnings as errors.
    Strict,
}

/// Result of validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Errors found.
    pub errors: Vec<ValidationError>,
    /// Warnings found.
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// Create a successful result.
    pub fn success() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Check if validation passed.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Check if validation passed with no warnings.
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty() && self.warnings.is_empty()
    }

    /// Add an error.
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// Add a warning.
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }
}

/// Validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error kind.
    pub kind: ValidationErrorKind,
    /// Location in IR.
    pub location: Option<ValidationLocation>,
    /// Error message.
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(loc) = &self.location {
            write!(f, "{}: {}: {}", loc, self.kind, self.message)
        } else {
            write!(f, "{}: {}", self.kind, self.message)
        }
    }
}

/// Error kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorKind {
    /// Type mismatch.
    TypeMismatch,
    /// Undefined value.
    UndefinedValue,
    /// Undefined block.
    UndefinedBlock,
    /// Unterminated block.
    UnterminatedBlock,
    /// Invalid operation.
    InvalidOperation,
    /// SSA violation.
    SsaViolation,
    /// Control flow error.
    ControlFlow,
    /// Missing entry block.
    MissingEntry,
}

impl std::fmt::Display for ValidationErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationErrorKind::TypeMismatch => write!(f, "type mismatch"),
            ValidationErrorKind::UndefinedValue => write!(f, "undefined value"),
            ValidationErrorKind::UndefinedBlock => write!(f, "undefined block"),
            ValidationErrorKind::UnterminatedBlock => write!(f, "unterminated block"),
            ValidationErrorKind::InvalidOperation => write!(f, "invalid operation"),
            ValidationErrorKind::SsaViolation => write!(f, "SSA violation"),
            ValidationErrorKind::ControlFlow => write!(f, "control flow error"),
            ValidationErrorKind::MissingEntry => write!(f, "missing entry block"),
        }
    }
}

/// Validation warning.
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning kind.
    pub kind: ValidationWarningKind,
    /// Location in IR.
    pub location: Option<ValidationLocation>,
    /// Warning message.
    pub message: String,
}

/// Warning kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationWarningKind {
    /// Unused value.
    UnusedValue,
    /// Unreachable code.
    UnreachableCode,
    /// Potential performance issue.
    Performance,
    /// Deprecated feature.
    Deprecated,
}

/// Location in IR for error reporting.
#[derive(Debug, Clone)]
pub struct ValidationLocation {
    /// Block ID.
    pub block: Option<BlockId>,
    /// Instruction index.
    pub instruction: Option<usize>,
    /// Value ID.
    pub value: Option<ValueId>,
}

impl std::fmt::Display for ValidationLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if let Some(block) = &self.block {
            parts.push(format!("block {}", block));
        }
        if let Some(inst) = &self.instruction {
            parts.push(format!("instruction {}", inst));
        }
        if let Some(value) = &self.value {
            parts.push(format!("value {}", value));
        }
        write!(f, "{}", parts.join(", "))
    }
}

/// IR validator.
pub struct Validator {
    level: ValidationLevel,
    result: ValidationResult,
    defined_values: HashSet<ValueId>,
    defined_blocks: HashSet<BlockId>,
}

impl Validator {
    /// Create a new validator.
    pub fn new(level: ValidationLevel) -> Self {
        Self {
            level,
            result: ValidationResult::success(),
            defined_values: HashSet::new(),
            defined_blocks: HashSet::new(),
        }
    }

    /// Validate a module.
    pub fn validate(mut self, module: &IrModule) -> ValidationResult {
        if self.level == ValidationLevel::None {
            return ValidationResult::success();
        }

        // Collect defined values and blocks
        self.collect_definitions(module);

        // Check entry block exists
        if !self.defined_blocks.contains(&module.entry_block) {
            self.result.add_error(ValidationError {
                kind: ValidationErrorKind::MissingEntry,
                location: None,
                message: "Module has no entry block".to_string(),
            });
        }

        // Validate each block
        for (block_id, block) in &module.blocks {
            self.validate_block(module, *block_id, block);
        }

        // Full validation includes type checking
        if self.level >= ValidationLevel::Full {
            self.validate_types(module);
        }

        self.result
    }

    fn collect_definitions(&mut self, module: &IrModule) {
        // Parameters define values
        for param in &module.parameters {
            self.defined_values.insert(param.value_id);
        }

        // Collect all values
        for value_id in module.values.keys() {
            self.defined_values.insert(*value_id);
        }

        // Collect all blocks
        for block_id in module.blocks.keys() {
            self.defined_blocks.insert(*block_id);
        }
    }

    fn validate_block(&mut self, module: &IrModule, block_id: BlockId, block: &Block) {
        // Check block is terminated
        if block.terminator.is_none() {
            self.result.add_error(ValidationError {
                kind: ValidationErrorKind::UnterminatedBlock,
                location: Some(ValidationLocation {
                    block: Some(block_id),
                    instruction: None,
                    value: None,
                }),
                message: format!("Block {} is not terminated", block.label),
            });
        }

        // Validate instructions
        for (idx, inst) in block.instructions.iter().enumerate() {
            self.validate_instruction(module, block_id, idx, &inst.node);
        }

        // Validate terminator
        if let Some(term) = &block.terminator {
            self.validate_terminator(block_id, term);
        }
    }

    fn validate_instruction(
        &mut self,
        _module: &IrModule,
        block_id: BlockId,
        idx: usize,
        node: &IrNode,
    ) {
        let location = ValidationLocation {
            block: Some(block_id),
            instruction: Some(idx),
            value: None,
        };

        // Check value references
        match node {
            IrNode::BinaryOp(_, lhs, rhs) => {
                self.check_value_defined(*lhs, &location);
                self.check_value_defined(*rhs, &location);
            }
            IrNode::UnaryOp(_, val) => {
                self.check_value_defined(*val, &location);
            }
            IrNode::Compare(_, lhs, rhs) => {
                self.check_value_defined(*lhs, &location);
                self.check_value_defined(*rhs, &location);
            }
            IrNode::Load(ptr) => {
                self.check_value_defined(*ptr, &location);
            }
            IrNode::Store(ptr, val) => {
                self.check_value_defined(*ptr, &location);
                self.check_value_defined(*val, &location);
            }
            IrNode::Select(cond, then_val, else_val) => {
                self.check_value_defined(*cond, &location);
                self.check_value_defined(*then_val, &location);
                self.check_value_defined(*else_val, &location);
            }
            IrNode::Phi(entries) => {
                for (pred_block, val) in entries {
                    self.check_block_defined(*pred_block, &location);
                    self.check_value_defined(*val, &location);
                }
            }
            _ => {}
        }
    }

    fn validate_terminator(&mut self, block_id: BlockId, term: &Terminator) {
        let location = ValidationLocation {
            block: Some(block_id),
            instruction: None,
            value: None,
        };

        match term {
            Terminator::Branch(target) => {
                self.check_block_defined(*target, &location);
            }
            Terminator::CondBranch(cond, then_block, else_block) => {
                self.check_value_defined(*cond, &location);
                self.check_block_defined(*then_block, &location);
                self.check_block_defined(*else_block, &location);
            }
            Terminator::Switch(val, default, cases) => {
                self.check_value_defined(*val, &location);
                self.check_block_defined(*default, &location);
                for (_, target) in cases {
                    self.check_block_defined(*target, &location);
                }
            }
            Terminator::Return(Some(val)) => {
                self.check_value_defined(*val, &location);
            }
            Terminator::Return(None) | Terminator::Unreachable => {}
        }
    }

    fn validate_types(&mut self, module: &IrModule) {
        for block in module.blocks.values() {
            for inst in &block.instructions {
                if let Err(msg) = self.check_instruction_types(module, &inst.node, &inst.result_type)
                {
                    self.result.add_error(ValidationError {
                        kind: ValidationErrorKind::TypeMismatch,
                        location: Some(ValidationLocation {
                            block: Some(block.id),
                            instruction: None,
                            value: Some(inst.result),
                        }),
                        message: msg,
                    });
                }
            }
        }
    }

    fn check_instruction_types(
        &self,
        module: &IrModule,
        node: &IrNode,
        _result_ty: &IrType,
    ) -> Result<(), String> {
        match node {
            IrNode::BinaryOp(_, lhs, rhs) => {
                let lhs_ty = self.get_value_type(module, *lhs);
                let rhs_ty = self.get_value_type(module, *rhs);
                if lhs_ty != rhs_ty {
                    return Err(format!(
                        "Binary operation operand types don't match: {} vs {}",
                        lhs_ty, rhs_ty
                    ));
                }
            }
            IrNode::Compare(_, lhs, rhs) => {
                let lhs_ty = self.get_value_type(module, *lhs);
                let rhs_ty = self.get_value_type(module, *rhs);
                if lhs_ty != rhs_ty {
                    return Err(format!(
                        "Comparison operand types don't match: {} vs {}",
                        lhs_ty, rhs_ty
                    ));
                }
            }
            IrNode::Load(ptr) => {
                let ptr_ty = self.get_value_type(module, *ptr);
                if !ptr_ty.is_ptr() {
                    return Err(format!("Load requires pointer type, got {}", ptr_ty));
                }
            }
            IrNode::Store(ptr, _val) => {
                let ptr_ty = self.get_value_type(module, *ptr);
                if !ptr_ty.is_ptr() {
                    return Err(format!("Store requires pointer type, got {}", ptr_ty));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn get_value_type(&self, module: &IrModule, id: ValueId) -> IrType {
        module
            .values
            .get(&id)
            .map(|v| v.ty.clone())
            .unwrap_or(IrType::Void)
    }

    fn check_value_defined(&mut self, id: ValueId, location: &ValidationLocation) {
        if !self.defined_values.contains(&id) {
            self.result.add_error(ValidationError {
                kind: ValidationErrorKind::UndefinedValue,
                location: Some(location.clone()),
                message: format!("Value {} is not defined", id),
            });
        }
    }

    fn check_block_defined(&mut self, id: BlockId, location: &ValidationLocation) {
        if !self.defined_blocks.contains(&id) {
            self.result.add_error(ValidationError {
                kind: ValidationErrorKind::UndefinedBlock,
                location: Some(location.clone()),
                message: format!("Block {} is not defined", id),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrBuilder;

    #[test]
    fn test_validation_success() {
        let mut builder = IrBuilder::new("test");
        builder.ret();
        let module = builder.build();

        let result = module.validate(ValidationLevel::Full);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_unterminated_block() {
        let module = IrModule::new("test");
        // Entry block has no terminator

        let result = Validator::new(ValidationLevel::Basic).validate(&module);
        assert!(!result.is_ok());
        assert!(result
            .errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::UnterminatedBlock));
    }

    #[test]
    fn test_validation_level_none() {
        let module = IrModule::new("test");
        // No terminator, but validation level is None

        let result = Validator::new(ValidationLevel::None).validate(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_result_display() {
        let error = ValidationError {
            kind: ValidationErrorKind::TypeMismatch,
            location: None,
            message: "expected i32".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("type mismatch"));
        assert!(display.contains("expected i32"));
    }
}
