//! DSL constraint validation for CUDA code generation.
//!
//! This module validates that Rust code conforms to the restricted DSL
//! that can be transpiled to CUDA.

use syn::visit::Visit;
use syn::{
    Expr, ExprAsync, ExprAwait, ExprClosure, ExprForLoop, ExprLoop, ExprWhile, ExprYield, ItemFn,
    Stmt,
};
use thiserror::Error;

/// Validation errors for DSL constraint violations.
#[derive(Error, Debug, Clone)]
pub enum ValidationError {
    /// Loops are not supported (use parallel threads instead).
    #[error("Loops are not supported in stencil kernels. Use parallel threads instead. Found: {0}")]
    LoopNotAllowed(String),

    /// Closures are not supported.
    #[error("Closures are not supported in stencil kernels. Found at: {0}")]
    ClosureNotAllowed(String),

    /// Async/await is not supported.
    #[error("Async/await is not supported in GPU kernels")]
    AsyncNotAllowed,

    /// Heap allocation is not supported.
    #[error("Heap allocation ({0}) is not supported in GPU kernels")]
    HeapAllocationNotAllowed(String),

    /// Unsupported expression type.
    #[error("Unsupported expression: {0}")]
    UnsupportedExpression(String),

    /// Recursion detected.
    #[error("Recursion is not supported in GPU kernels")]
    RecursionNotAllowed,

    /// Invalid function signature.
    #[error("Invalid function signature: {0}")]
    InvalidSignature(String),
}

/// Visitor that checks for DSL constraint violations.
struct DslValidator {
    errors: Vec<ValidationError>,
    function_name: String,
}

impl DslValidator {
    fn new(function_name: String) -> Self {
        Self {
            errors: Vec::new(),
            function_name,
        }
    }

    fn check_heap_allocation(&mut self, expr: &Expr) {
        // Check for common heap-allocating patterns
        if let Expr::Call(call) = expr {
            if let Expr::Path(path) = call.func.as_ref() {
                let path_str = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");

                // Check for Vec, Box, String, etc.
                let heap_types = ["Vec", "Box", "String", "Rc", "Arc", "HashMap", "HashSet"];
                for heap_type in &heap_types {
                    if path_str.contains(heap_type) {
                        self.errors
                            .push(ValidationError::HeapAllocationNotAllowed(path_str.clone()));
                        return;
                    }
                }
            }
        }

        // Check for vec! macro
        if let Expr::Macro(mac) = expr {
            let macro_name = mac.mac.path.segments.last().map(|s| s.ident.to_string());
            if macro_name == Some("vec".to_string()) {
                self.errors
                    .push(ValidationError::HeapAllocationNotAllowed("vec!".to_string()));
            }
        }
    }

    fn check_recursion(&mut self, expr: &Expr) {
        // Check for potential recursive calls
        if let Expr::Call(call) = expr {
            if let Expr::Path(path) = call.func.as_ref() {
                if let Some(segment) = path.path.segments.last() {
                    if segment.ident == self.function_name {
                        self.errors.push(ValidationError::RecursionNotAllowed);
                    }
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for DslValidator {
    fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
        self.errors.push(ValidationError::LoopNotAllowed(
            "for loop".to_string(),
        ));
        // Still visit children to find more errors
        syn::visit::visit_expr_for_loop(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast ExprWhile) {
        self.errors.push(ValidationError::LoopNotAllowed(
            "while loop".to_string(),
        ));
        syn::visit::visit_expr_while(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast ExprLoop) {
        self.errors.push(ValidationError::LoopNotAllowed(
            "loop".to_string(),
        ));
        syn::visit::visit_expr_loop(self, node);
    }

    fn visit_expr_closure(&mut self, node: &'ast ExprClosure) {
        self.errors.push(ValidationError::ClosureNotAllowed(
            "closure expression".to_string(),
        ));
        syn::visit::visit_expr_closure(self, node);
    }

    fn visit_expr_async(&mut self, _node: &'ast ExprAsync) {
        self.errors.push(ValidationError::AsyncNotAllowed);
    }

    fn visit_expr_await(&mut self, _node: &'ast ExprAwait) {
        self.errors.push(ValidationError::AsyncNotAllowed);
    }

    fn visit_expr_yield(&mut self, _node: &'ast ExprYield) {
        self.errors.push(ValidationError::UnsupportedExpression(
            "yield expression".to_string(),
        ));
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        // Check for heap allocations
        self.check_heap_allocation(node);

        // Check for recursion
        self.check_recursion(node);

        // Continue visiting children
        syn::visit::visit_expr(self, node);
    }
}

/// Validate that a function conforms to the stencil kernel DSL.
///
/// # Constraints
///
/// - No loops (`for`, `while`, `loop`) - use parallel threads
/// - No closures
/// - No async/await
/// - No heap allocations (Vec, Box, String, etc.)
/// - No recursion
/// - No trait objects
///
/// # Returns
///
/// `Ok(())` if validation passes, `Err` with the first validation error otherwise.
pub fn validate_function(func: &ItemFn) -> Result<(), ValidationError> {
    let function_name = func.sig.ident.to_string();
    let mut validator = DslValidator::new(function_name);

    // Visit all statements in the function body
    for stmt in &func.block.stmts {
        match stmt {
            Stmt::Expr(expr, _) => {
                validator.visit_expr(expr);
            }
            Stmt::Local(syn::Local { init: Some(syn::LocalInit { expr, .. }), .. }) => {
                validator.visit_expr(expr);
            }
            _ => {}
        }
    }

    // Also visit the entire block to catch nested constructs
    syn::visit::visit_block(&mut validator, &func.block);

    // Return first error if any
    if let Some(error) = validator.errors.into_iter().next() {
        Err(error)
    } else {
        Ok(())
    }
}

/// Validate function signature for stencil kernels.
///
/// Ensures the function has appropriate parameter types and
/// returns void or a simple type.
pub fn validate_stencil_signature(func: &ItemFn) -> Result<(), ValidationError> {
    // Check that the function is not async
    if func.sig.asyncness.is_some() {
        return Err(ValidationError::AsyncNotAllowed);
    }

    // Check that the function doesn't use generics (for now)
    if !func.sig.generics.params.is_empty() {
        return Err(ValidationError::InvalidSignature(
            "Generic parameters are not supported in stencil kernels".to_string(),
        ));
    }

    Ok(())
}

/// Check if a statement might need special handling.
pub fn is_simple_assignment(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Local(_) | Stmt::Expr(Expr::Assign(_), _))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_valid_function() {
        let func: ItemFn = parse_quote! {
            fn add(a: f32, b: f32) -> f32 {
                let c = a + b;
                c * 2.0
            }
        };

        assert!(validate_function(&func).is_ok());
    }

    #[test]
    fn test_for_loop_rejected() {
        let func: ItemFn = parse_quote! {
            fn sum(data: &[f32]) -> f32 {
                let mut total = 0.0;
                for x in data {
                    total += x;
                }
                total
            }
        };

        let result = validate_function(&func);
        assert!(matches!(result, Err(ValidationError::LoopNotAllowed(_))));
    }

    #[test]
    fn test_while_loop_rejected() {
        let func: ItemFn = parse_quote! {
            fn countdown(n: i32) -> i32 {
                let mut i = n;
                while i > 0 {
                    i -= 1;
                }
                i
            }
        };

        let result = validate_function(&func);
        assert!(matches!(result, Err(ValidationError::LoopNotAllowed(_))));
    }

    #[test]
    fn test_closure_rejected() {
        let func: ItemFn = parse_quote! {
            fn apply(x: f32) -> f32 {
                let f = |v| v * 2.0;
                f(x)
            }
        };

        let result = validate_function(&func);
        assert!(matches!(result, Err(ValidationError::ClosureNotAllowed(_))));
    }

    #[test]
    fn test_if_else_allowed() {
        let func: ItemFn = parse_quote! {
            fn clamp(x: f32, min: f32, max: f32) -> f32 {
                if x < min {
                    min
                } else if x > max {
                    max
                } else {
                    x
                }
            }
        };

        assert!(validate_function(&func).is_ok());
    }

    #[test]
    fn test_stencil_pattern_allowed() {
        let func: ItemFn = parse_quote! {
            fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
                let curr = p[pos.idx()];
                let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
                p_prev[pos.idx()] = 2.0 * curr - p_prev[pos.idx()] + c2 * lap;
            }
        };

        assert!(validate_function(&func).is_ok());
    }

    #[test]
    fn test_async_rejected() {
        let func: ItemFn = parse_quote! {
            async fn fetch(x: f32) -> f32 {
                x
            }
        };

        let result = validate_stencil_signature(&func);
        assert!(matches!(result, Err(ValidationError::AsyncNotAllowed)));
    }
}
