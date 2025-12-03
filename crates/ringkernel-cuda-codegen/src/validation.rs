//! DSL constraint validation for CUDA code generation.
//!
//! This module validates that Rust code conforms to the restricted DSL
//! that can be transpiled to CUDA.
//!
//! # Validation Modes
//!
//! Different kernel types have different validation requirements:
//!
//! - **Stencil**: No loops allowed (use parallel threads)
//! - **Generic**: Loops allowed for general CUDA kernels
//! - **RingKernel**: Loops required for persistent actor kernels

use syn::visit::Visit;
use syn::{
    Expr, ExprAsync, ExprAwait, ExprClosure, ExprForLoop, ExprLoop, ExprWhile, ExprYield, ItemFn,
    Stmt,
};
use thiserror::Error;

/// Validation mode for different kernel types.
///
/// Different kernel patterns have different requirements for what
/// Rust constructs are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Stencil kernels: No loops allowed (current behavior).
    /// Work is parallelized across threads, so loops would serialize work.
    #[default]
    Stencil,

    /// Generic CUDA kernels: Loops allowed.
    /// Useful for kernels that need sequential processing within a thread.
    Generic,

    /// Ring/Actor kernels: Loops required (persistent message loop).
    /// These kernels run persistently and process messages in a loop.
    RingKernel,
}

impl ValidationMode {
    /// Check if loops are allowed in this validation mode.
    pub fn allows_loops(&self) -> bool {
        matches!(self, ValidationMode::Generic | ValidationMode::RingKernel)
    }

    /// Check if loops are required in this validation mode.
    pub fn requires_loops(&self) -> bool {
        matches!(self, ValidationMode::RingKernel)
    }
}

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

    /// Missing required loop for ring kernels.
    #[error("Ring kernels require a message processing loop")]
    LoopRequired,
}

/// Visitor that checks for DSL constraint violations.
struct DslValidator {
    errors: Vec<ValidationError>,
    function_name: String,
    mode: ValidationMode,
    loop_count: usize,
}

impl DslValidator {
    fn with_mode(function_name: String, mode: ValidationMode) -> Self {
        Self {
            errors: Vec::new(),
            function_name,
            mode,
            loop_count: 0,
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
        if !self.mode.allows_loops() {
            self.errors.push(ValidationError::LoopNotAllowed(
                "for loop".to_string(),
            ));
        }
        self.loop_count += 1;
        // Still visit children to find more errors
        syn::visit::visit_expr_for_loop(self, node);
    }

    fn visit_expr_while(&mut self, node: &'ast ExprWhile) {
        if !self.mode.allows_loops() {
            self.errors.push(ValidationError::LoopNotAllowed(
                "while loop".to_string(),
            ));
        }
        self.loop_count += 1;
        syn::visit::visit_expr_while(self, node);
    }

    fn visit_expr_loop(&mut self, node: &'ast ExprLoop) {
        if !self.mode.allows_loops() {
            self.errors.push(ValidationError::LoopNotAllowed(
                "loop".to_string(),
            ));
        }
        self.loop_count += 1;
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
    validate_function_with_mode(func, ValidationMode::Stencil)
}

/// Validate a function with a specific validation mode.
///
/// Different kernel types have different validation requirements:
///
/// - `ValidationMode::Stencil`: No loops allowed (current behavior for stencil kernels)
/// - `ValidationMode::Generic`: Loops allowed for general CUDA kernels
/// - `ValidationMode::RingKernel`: Loops required for persistent actor kernels
///
/// # Arguments
///
/// * `func` - The function to validate
/// * `mode` - The validation mode to use
///
/// # Returns
///
/// `Ok(())` if validation passes, `Err` with the first validation error otherwise.
///
/// # Example
///
/// ```ignore
/// use ringkernel_cuda_codegen::{validate_function_with_mode, ValidationMode};
/// use syn::parse_quote;
///
/// // Generic kernel with loops allowed
/// let func: syn::ItemFn = parse_quote! {
///     fn process(data: &mut [f32], n: i32) {
///         for i in 0..n {
///             data[i as usize] = data[i as usize] * 2.0;
///         }
///     }
/// };
///
/// // This would fail with Stencil mode, but passes with Generic mode
/// assert!(validate_function_with_mode(&func, ValidationMode::Generic).is_ok());
/// ```
pub fn validate_function_with_mode(
    func: &ItemFn,
    mode: ValidationMode,
) -> Result<(), ValidationError> {
    let function_name = func.sig.ident.to_string();
    let mut validator = DslValidator::with_mode(function_name, mode);

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

    // Check if loops are required but none found
    if mode.requires_loops() && validator.loop_count == 0 {
        return Err(ValidationError::LoopRequired);
    }

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

    // === ValidationMode tests ===

    #[test]
    fn test_generic_mode_allows_for_loop() {
        let func: ItemFn = parse_quote! {
            fn process(data: &mut [f32], n: i32) {
                for i in 0..n {
                    data[i as usize] = data[i as usize] * 2.0;
                }
            }
        };

        // Stencil mode rejects loops
        assert!(matches!(
            validate_function_with_mode(&func, ValidationMode::Stencil),
            Err(ValidationError::LoopNotAllowed(_))
        ));

        // Generic mode allows loops
        assert!(validate_function_with_mode(&func, ValidationMode::Generic).is_ok());
    }

    #[test]
    fn test_generic_mode_allows_while_loop() {
        let func: ItemFn = parse_quote! {
            fn process(data: &mut [f32]) {
                let mut i = 0;
                while i < 10 {
                    data[i] = 0.0;
                    i += 1;
                }
            }
        };

        // Generic mode allows while loops
        assert!(validate_function_with_mode(&func, ValidationMode::Generic).is_ok());
    }

    #[test]
    fn test_generic_mode_allows_infinite_loop() {
        let func: ItemFn = parse_quote! {
            fn process(active: &u32) {
                loop {
                    if *active == 0 {
                        return;
                    }
                }
            }
        };

        // Generic mode allows infinite loops
        assert!(validate_function_with_mode(&func, ValidationMode::Generic).is_ok());
    }

    #[test]
    fn test_ring_kernel_mode_requires_loop() {
        let func_no_loop: ItemFn = parse_quote! {
            fn handler(x: f32) -> f32 {
                x * 2.0
            }
        };

        // RingKernel mode requires at least one loop
        assert!(matches!(
            validate_function_with_mode(&func_no_loop, ValidationMode::RingKernel),
            Err(ValidationError::LoopRequired)
        ));

        let func_with_loop: ItemFn = parse_quote! {
            fn handler(active: &u32) {
                while *active != 0 {
                    // Process messages
                }
            }
        };

        // RingKernel mode accepts function with loop
        assert!(validate_function_with_mode(&func_with_loop, ValidationMode::RingKernel).is_ok());
    }

    #[test]
    fn test_validation_mode_allows_loops() {
        assert!(!ValidationMode::Stencil.allows_loops());
        assert!(ValidationMode::Generic.allows_loops());
        assert!(ValidationMode::RingKernel.allows_loops());
    }

    #[test]
    fn test_validation_mode_requires_loops() {
        assert!(!ValidationMode::Stencil.requires_loops());
        assert!(!ValidationMode::Generic.requires_loops());
        assert!(ValidationMode::RingKernel.requires_loops());
    }

    #[test]
    fn test_closures_rejected_in_all_modes() {
        // Test without loop (for Stencil and Generic modes)
        let func: ItemFn = parse_quote! {
            fn apply(x: f32) -> f32 {
                let f = |v| v * 2.0;
                f(x)
            }
        };

        // Closures are rejected in Stencil and Generic modes
        assert!(matches!(
            validate_function_with_mode(&func, ValidationMode::Stencil),
            Err(ValidationError::ClosureNotAllowed(_))
        ));
        assert!(matches!(
            validate_function_with_mode(&func, ValidationMode::Generic),
            Err(ValidationError::ClosureNotAllowed(_))
        ));

        // For RingKernel mode, we need a function with a loop that also has a closure
        let func_with_loop: ItemFn = parse_quote! {
            fn apply(x: f32) -> f32 {
                loop {
                    let f = |v| v * 2.0;
                    if f(x) > 0.0 { break; }
                }
                x
            }
        };

        // Closures are rejected in RingKernel mode too
        assert!(matches!(
            validate_function_with_mode(&func_with_loop, ValidationMode::RingKernel),
            Err(ValidationError::ClosureNotAllowed(_))
        ));
    }
}
