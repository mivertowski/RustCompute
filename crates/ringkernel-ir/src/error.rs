//! IR error types.

use thiserror::Error;

use crate::{BlockId, IrType, ValueId};

/// IR result type.
pub type IrResult<T> = Result<T, IrError>;

/// IR errors.
#[derive(Debug, Error)]
pub enum IrError {
    /// Type mismatch.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type.
        expected: IrType,
        /// Actual type.
        actual: IrType,
    },

    /// Undefined value reference.
    #[error("Undefined value: {0}")]
    UndefinedValue(ValueId),

    /// Undefined block reference.
    #[error("Undefined block: {0}")]
    UndefinedBlock(BlockId),

    /// Block not terminated.
    #[error("Block {0} is not terminated")]
    UnterminatedBlock(BlockId),

    /// Invalid operation for type.
    #[error("Invalid operation {op} for type {ty}")]
    InvalidOperation {
        /// Operation name.
        op: String,
        /// Type involved.
        ty: IrType,
    },

    /// Invalid cast.
    #[error("Cannot cast from {from} to {to}")]
    InvalidCast {
        /// Source type.
        from: IrType,
        /// Target type.
        to: IrType,
    },

    /// Capability not supported.
    #[error("Capability not supported: {0}")]
    CapabilityNotSupported(String),

    /// Invalid parameter count.
    #[error("Expected {expected} parameters, got {actual}")]
    ParameterCountMismatch {
        /// Expected count.
        expected: usize,
        /// Actual count.
        actual: usize,
    },

    /// Invalid vector size.
    #[error("Invalid vector size: {0} (must be 2, 3, or 4)")]
    InvalidVectorSize(u8),

    /// Invalid array size.
    #[error("Invalid array size: {0}")]
    InvalidArraySize(usize),

    /// Missing entry block.
    #[error("Module has no entry block")]
    MissingEntryBlock,

    /// Duplicate definition.
    #[error("Duplicate definition: {0}")]
    DuplicateDefinition(String),

    /// Invalid phi node.
    #[error("Invalid phi node: {0}")]
    InvalidPhi(String),

    /// Control flow error.
    #[error("Control flow error: {0}")]
    ControlFlowError(String),

    /// Validation error.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = IrError::TypeMismatch {
            expected: IrType::I32,
            actual: IrType::F32,
        };
        assert!(err.to_string().contains("i32"));
        assert!(err.to_string().contains("f32"));
    }

    #[test]
    fn test_undefined_value() {
        let id = ValueId::new();
        let err = IrError::UndefinedValue(id);
        assert!(err.to_string().contains("Undefined value"));
    }
}
