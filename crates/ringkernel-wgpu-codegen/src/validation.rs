//! DSL validation for WGSL code generation.
//!
//! This module provides validation for the Rust DSL subset that can be
//! transpiled to WGSL.

use thiserror::Error;

/// Errors that can occur during validation.
#[derive(Error, Debug)]
pub enum ValidationError {
    /// Unsupported Rust construct.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Invalid DSL usage.
    #[error("Invalid DSL usage: {0}")]
    InvalidDsl(String),

    /// Loop not allowed in this mode.
    #[error("Loops not allowed in {0} mode")]
    LoopNotAllowed(String),
}

/// Validation mode determines which constructs are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Stencil kernels - classic grid-based computation.
    /// Loops are forbidden (each thread handles one cell).
    Stencil,

    /// Generic kernels - general-purpose compute shaders.
    /// Loops are allowed.
    #[default]
    Generic,

    /// Ring kernels - persistent message processing.
    /// Loops are required (the persistent message loop).
    RingKernel,
}

impl ValidationMode {
    /// Check if this mode allows loops.
    pub fn allows_loops(&self) -> bool {
        match self {
            ValidationMode::Stencil => false,
            ValidationMode::Generic => true,
            ValidationMode::RingKernel => true,
        }
    }

    /// Check if this mode requires loops.
    pub fn requires_loops(&self) -> bool {
        matches!(self, ValidationMode::RingKernel)
    }
}

/// Validate a function for WGSL transpilation.
pub fn validate_function(func: &syn::ItemFn) -> Result<(), ValidationError> {
    validate_function_with_mode(func, ValidationMode::default())
}

/// Validate a function with a specific validation mode.
pub fn validate_function_with_mode(
    func: &syn::ItemFn,
    mode: ValidationMode,
) -> Result<(), ValidationError> {
    // Check for unsupported attributes
    for attr in &func.attrs {
        let path = attr.path().to_token_stream().to_string();
        if path != "doc" && path != "allow" && path != "cfg" {
            return Err(ValidationError::Unsupported(format!(
                "Function attribute: {}",
                path
            )));
        }
    }

    // Check for async functions (not supported)
    if func.sig.asyncness.is_some() {
        return Err(ValidationError::Unsupported(
            "Async functions are not supported in WGSL".to_string(),
        ));
    }

    // Check for generics (not supported)
    if !func.sig.generics.params.is_empty() {
        return Err(ValidationError::Unsupported(
            "Generic functions are not supported in WGSL".to_string(),
        ));
    }

    // Check for variadic functions (not supported)
    if func.sig.variadic.is_some() {
        return Err(ValidationError::Unsupported(
            "Variadic functions are not supported in WGSL".to_string(),
        ));
    }

    // Validate the function body
    validate_block(&func.block, mode)?;

    Ok(())
}

/// Validate a block of statements.
fn validate_block(block: &syn::Block, mode: ValidationMode) -> Result<(), ValidationError> {
    for stmt in &block.stmts {
        validate_stmt(stmt, mode)?;
    }
    Ok(())
}

/// Validate a single statement.
fn validate_stmt(stmt: &syn::Stmt, mode: ValidationMode) -> Result<(), ValidationError> {
    match stmt {
        syn::Stmt::Local(local) => {
            // Validate the initializer if present
            if let Some(init) = &local.init {
                validate_expr(&init.expr, mode)?;
            }
            Ok(())
        }
        syn::Stmt::Expr(expr, _) => validate_expr(expr, mode),
        syn::Stmt::Item(_) => Err(ValidationError::Unsupported(
            "Nested items are not supported".to_string(),
        )),
        syn::Stmt::Macro(_) => Err(ValidationError::Unsupported(
            "Macros are not supported in WGSL DSL".to_string(),
        )),
    }
}

/// Validate an expression.
fn validate_expr(expr: &syn::Expr, mode: ValidationMode) -> Result<(), ValidationError> {
    match expr {
        // Simple expressions - always allowed
        syn::Expr::Lit(_) => Ok(()),
        syn::Expr::Path(_) => Ok(()),
        syn::Expr::Paren(p) => validate_expr(&p.expr, mode),

        // Binary and unary operations
        syn::Expr::Binary(bin) => {
            validate_expr(&bin.left, mode)?;
            validate_expr(&bin.right, mode)
        }
        syn::Expr::Unary(unary) => validate_expr(&unary.expr, mode),

        // Array indexing
        syn::Expr::Index(idx) => {
            validate_expr(&idx.expr, mode)?;
            validate_expr(&idx.index, mode)
        }

        // Function and method calls
        syn::Expr::Call(call) => {
            validate_expr(&call.func, mode)?;
            for arg in &call.args {
                validate_expr(arg, mode)?;
            }
            Ok(())
        }
        syn::Expr::MethodCall(method) => {
            validate_expr(&method.receiver, mode)?;
            for arg in &method.args {
                validate_expr(arg, mode)?;
            }
            Ok(())
        }

        // Control flow
        syn::Expr::If(if_expr) => {
            validate_expr(&if_expr.cond, mode)?;
            validate_block(&if_expr.then_branch, mode)?;
            if let Some((_, else_branch)) = &if_expr.else_branch {
                validate_expr(else_branch, mode)?;
            }
            Ok(())
        }
        syn::Expr::Block(block) => validate_block(&block.block, mode),

        // Loops - mode-dependent
        syn::Expr::ForLoop(for_loop) => {
            if !mode.allows_loops() {
                return Err(ValidationError::LoopNotAllowed("stencil".to_string()));
            }
            validate_expr(&for_loop.expr, mode)?;
            validate_block(&for_loop.body, mode)
        }
        syn::Expr::While(while_loop) => {
            if !mode.allows_loops() {
                return Err(ValidationError::LoopNotAllowed("stencil".to_string()));
            }
            validate_expr(&while_loop.cond, mode)?;
            validate_block(&while_loop.body, mode)
        }
        syn::Expr::Loop(loop_expr) => {
            if !mode.allows_loops() {
                return Err(ValidationError::LoopNotAllowed("stencil".to_string()));
            }
            validate_block(&loop_expr.body, mode)
        }

        // Return and control
        syn::Expr::Return(ret) => {
            if let Some(expr) = &ret.expr {
                validate_expr(expr, mode)?;
            }
            Ok(())
        }
        syn::Expr::Break(_) => Ok(()),
        syn::Expr::Continue(_) => Ok(()),

        // Assignments
        syn::Expr::Assign(assign) => {
            validate_expr(&assign.left, mode)?;
            validate_expr(&assign.right, mode)
        }

        // Type casts
        syn::Expr::Cast(cast) => validate_expr(&cast.expr, mode),

        // Struct literals
        syn::Expr::Struct(struct_expr) => {
            for field in &struct_expr.fields {
                validate_expr(&field.expr, mode)?;
            }
            Ok(())
        }

        // Field access
        syn::Expr::Field(field) => validate_expr(&field.base, mode),

        // Match expressions
        syn::Expr::Match(match_expr) => {
            validate_expr(&match_expr.expr, mode)?;
            for arm in &match_expr.arms {
                validate_expr(&arm.body, mode)?;
            }
            Ok(())
        }

        // Range expressions (for loops)
        syn::Expr::Range(range) => {
            if let Some(start) = &range.start {
                validate_expr(start, mode)?;
            }
            if let Some(end) = &range.end {
                validate_expr(end, mode)?;
            }
            Ok(())
        }

        // Reference and dereference
        syn::Expr::Reference(ref_expr) => validate_expr(&ref_expr.expr, mode),

        // Unsupported expressions
        _ => Err(ValidationError::Unsupported(format!(
            "Expression type: {}",
            quote::ToTokens::to_token_stream(expr)
        ))),
    }
}

use quote::ToTokens;

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_simple_function_validates() {
        let func: syn::ItemFn = parse_quote! {
            fn add(a: f32, b: f32) -> f32 {
                a + b
            }
        };
        assert!(validate_function(&func).is_ok());
    }

    #[test]
    fn test_async_function_rejected() {
        let func: syn::ItemFn = parse_quote! {
            async fn process() {}
        };
        assert!(validate_function(&func).is_err());
    }

    #[test]
    fn test_generic_function_rejected() {
        let func: syn::ItemFn = parse_quote! {
            fn generic<T>(x: T) -> T { x }
        };
        assert!(validate_function(&func).is_err());
    }

    #[test]
    fn test_stencil_mode_rejects_loops() {
        let func: syn::ItemFn = parse_quote! {
            fn with_loop() {
                for i in 0..10 {
                    process(i);
                }
            }
        };
        assert!(validate_function_with_mode(&func, ValidationMode::Stencil).is_err());
    }

    #[test]
    fn test_generic_mode_allows_loops() {
        let func: syn::ItemFn = parse_quote! {
            fn with_loop() {
                for i in 0..10 {
                    process(i);
                }
            }
        };
        assert!(validate_function_with_mode(&func, ValidationMode::Generic).is_ok());
    }
}
