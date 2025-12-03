//! Core Rust-to-CUDA transpiler.
//!
//! This module handles the translation of Rust AST to CUDA C code.

use crate::handler::{ContextMethod, HandlerSignature};
use crate::intrinsics::{IntrinsicRegistry, RingKernelIntrinsic, StencilIntrinsic};
use crate::loops::{extract_loop_var, RangeInfo};
use crate::shared::{rust_to_cuda_element_type, SharedMemoryConfig, SharedMemoryDecl};
use crate::stencil::StencilConfig;
use crate::types::{is_grid_pos_type, is_ring_context_type, TypeMapper};
use crate::validation::ValidationMode;
use crate::{Result, TranspileError};
use quote::ToTokens;
use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprBreak, ExprCall, ExprCast, ExprContinue, ExprForLoop,
    ExprIf, ExprIndex, ExprLit, ExprLoop, ExprMatch, ExprMethodCall, ExprParen, ExprPath,
    ExprReturn, ExprStruct, ExprUnary, ExprWhile, FnArg, ItemFn, Lit, Pat, ReturnType, Stmt, UnOp,
};

/// CUDA code transpiler.
pub struct CudaTranspiler {
    /// Stencil configuration (if generating a stencil kernel).
    config: Option<StencilConfig>,
    /// Type mapper.
    type_mapper: TypeMapper,
    /// Intrinsic registry.
    intrinsics: IntrinsicRegistry,
    /// Variables known to be the GridPos context.
    grid_pos_vars: Vec<String>,
    /// Variables known to be RingContext references.
    context_vars: Vec<String>,
    /// Current indentation level.
    indent: usize,
    /// Validation mode for loop handling.
    validation_mode: ValidationMode,
    /// Shared memory configuration.
    shared_memory: SharedMemoryConfig,
    /// Variables that are SharedTile or SharedArray types.
    pub shared_vars: std::collections::HashMap<String, SharedVarInfo>,
    /// Whether we're in ring kernel mode (enables context method inlining).
    ring_kernel_mode: bool,
}

/// Information about a shared memory variable.
#[derive(Debug, Clone)]
pub struct SharedVarInfo {
    /// Variable name.
    pub name: String,
    /// Whether it's a 2D tile (true) or 1D array (false).
    pub is_tile: bool,
    /// Dimensions: [size] for 1D, [height, width] for 2D.
    pub dimensions: Vec<usize>,
    /// Element type (CUDA type string).
    pub element_type: String,
}

impl CudaTranspiler {
    /// Create a new transpiler with stencil configuration.
    pub fn new(config: StencilConfig) -> Self {
        Self {
            config: Some(config),
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 1, // Start with 1 level for function body
            validation_mode: ValidationMode::Stencil,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: std::collections::HashMap::new(),
            ring_kernel_mode: false,
        }
    }

    /// Create a new transpiler without stencil configuration.
    pub fn new_generic() -> Self {
        Self {
            config: None,
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 1,
            validation_mode: ValidationMode::Generic,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: std::collections::HashMap::new(),
            ring_kernel_mode: false,
        }
    }

    /// Create a new transpiler with a specific validation mode.
    pub fn with_mode(mode: ValidationMode) -> Self {
        Self {
            config: None,
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 1,
            validation_mode: mode,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: std::collections::HashMap::new(),
            ring_kernel_mode: false,
        }
    }

    /// Create a transpiler configured for ring kernel handler transpilation.
    pub fn for_ring_kernel() -> Self {
        Self {
            config: None,
            type_mapper: crate::types::ring_kernel_type_mapper(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 2, // Inside kernel + loop
            validation_mode: ValidationMode::Generic,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: std::collections::HashMap::new(),
            ring_kernel_mode: true,
        }
    }

    /// Set the validation mode.
    pub fn set_validation_mode(&mut self, mode: ValidationMode) {
        self.validation_mode = mode;
    }

    /// Get the shared memory configuration.
    pub fn shared_memory(&self) -> &SharedMemoryConfig {
        &self.shared_memory
    }

    /// Get current indentation string.
    fn indent_str(&self) -> String {
        "    ".repeat(self.indent)
    }

    /// Transpile a stencil kernel function.
    pub fn transpile_stencil(&mut self, func: &ItemFn) -> Result<String> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| TranspileError::Unsupported("No stencil config provided".into()))?
            .clone();

        // Identify GridPos parameters
        for param in &func.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                if is_grid_pos_type(&pat_type.ty) {
                    if let Pat::Ident(ident) = pat_type.pat.as_ref() {
                        self.grid_pos_vars.push(ident.ident.to_string());
                    }
                }
            }
        }

        // Generate function signature (without GridPos params)
        let signature = self.transpile_kernel_signature(func)?;

        // Generate preamble (thread index calculations)
        let preamble = config.generate_preamble();

        // Generate function body
        let body = self.transpile_block(&func.block)?;

        Ok(format!(
            "extern \"C\" __global__ void {signature} {{\n{preamble}\n{body}}}\n"
        ))
    }

    /// Transpile a generic (non-stencil) kernel function.
    ///
    /// This generates a `__global__` kernel without stencil-specific preamble.
    /// The kernel code can use `thread_idx_x()`, `block_idx_x()` etc. to access
    /// CUDA thread indices directly.
    pub fn transpile_generic_kernel(&mut self, func: &ItemFn) -> Result<String> {
        // Generate function signature with all params
        let signature = self.transpile_generic_kernel_signature(func)?;

        // Generate function body
        let body = self.transpile_block(&func.block)?;

        Ok(format!(
            "extern \"C\" __global__ void {signature} {{\n{body}}}\n"
        ))
    }

    /// Transpile a handler function into a persistent ring kernel.
    ///
    /// This wraps the handler body in a persistent message-processing loop
    /// with control block integration, queue operations, and HLC support.
    pub fn transpile_ring_kernel(
        &mut self,
        handler: &ItemFn,
        config: &crate::ring_kernel::RingKernelConfig,
    ) -> Result<String> {
        use std::fmt::Write;

        // Parse handler signature
        let handler_sig = HandlerSignature::parse(handler, &self.type_mapper)?;

        // Track context variables for method inlining
        for param in &handler.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                if is_ring_context_type(&pat_type.ty) {
                    if let Pat::Ident(ident) = pat_type.pat.as_ref() {
                        self.context_vars.push(ident.ident.to_string());
                    }
                }
            }
        }

        // Enable ring kernel mode for context method inlining
        self.ring_kernel_mode = true;

        let mut output = String::new();

        // Generate struct definitions
        output.push_str(&crate::ring_kernel::generate_control_block_struct());
        output.push('\n');

        if config.enable_hlc {
            output.push_str(&crate::ring_kernel::generate_hlc_struct());
            output.push('\n');
        }

        if config.enable_k2k {
            output.push_str(&crate::ring_kernel::generate_k2k_structs());
            output.push('\n');
        }

        // Generate message/response struct definitions if needed
        if let Some(ref msg_param) = handler_sig.message_param {
            // Extract type name from the parameter
            let type_name = msg_param.rust_type
                .trim_start_matches('&')
                .trim_start_matches("mut ")
                .trim();
            if !type_name.is_empty() && type_name != "f32" && type_name != "i32" {
                writeln!(output, "// Message type: {}", type_name).unwrap();
            }
        }

        if let Some(ref ret_type) = handler_sig.return_type {
            if ret_type.is_struct {
                writeln!(output, "// Response type: {}", ret_type.rust_type).unwrap();
            }
        }

        // Generate kernel signature
        output.push_str(&config.generate_signature());
        output.push_str(" {\n");

        // Generate preamble
        output.push_str(&config.generate_preamble("    "));

        // Generate message loop header
        output.push_str(&config.generate_loop_header("    "));

        // Generate message deserialization if handler has message param
        if let Some(ref msg_param) = handler_sig.message_param {
            let type_name = msg_param.rust_type
                .trim_start_matches('&')
                .trim_start_matches("mut ")
                .trim();
            if !type_name.is_empty() {
                writeln!(output, "        // Message deserialization").unwrap();
                writeln!(output, "        // {}* {} = ({}*)msg_ptr;",
                    type_name, msg_param.name, type_name).unwrap();
                output.push('\n');
            }
        }

        // Transpile handler body
        self.indent = 2; // Inside the message loop
        let handler_body = self.transpile_block(&handler.block)?;

        // Insert handler code with proper indentation
        writeln!(output, "        // === USER HANDLER CODE ===").unwrap();
        for line in handler_body.lines() {
            if !line.trim().is_empty() {
                // Add extra indent for being inside the loop
                writeln!(output, "    {}", line).unwrap();
            }
        }
        writeln!(output, "        // === END HANDLER CODE ===").unwrap();

        // Generate response serialization if handler returns a value
        if let Some(ref ret_type) = handler_sig.return_type {
            writeln!(output).unwrap();
            writeln!(output, "        // Response serialization").unwrap();
            if ret_type.is_struct {
                writeln!(output, "        // memcpy(&output_buffer[_out_idx * RESP_SIZE], &response, sizeof({}));",
                    ret_type.cuda_type).unwrap();
            }
        }

        // Generate message completion
        output.push_str(&config.generate_message_complete("    "));

        // Generate loop footer
        output.push_str(&config.generate_loop_footer("    "));

        // Generate epilogue
        output.push_str(&config.generate_epilogue("    "));

        output.push_str("}\n");

        Ok(output)
    }

    /// Transpile a generic kernel function signature (keeps all params).
    fn transpile_generic_kernel_signature(&self, func: &ItemFn) -> Result<String> {
        let name = func.sig.ident.to_string();

        let mut params = Vec::new();
        for param in &func.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                let param_name = match pat_type.pat.as_ref() {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    _ => {
                        return Err(TranspileError::Unsupported(
                            "Complex pattern in parameter".into(),
                        ))
                    }
                };

                let cuda_type = self.type_mapper.map_type(&pat_type.ty)?;
                params.push(format!("{} {}", cuda_type.to_cuda_string(), param_name));
            }
        }

        Ok(format!("{}({})", name, params.join(", ")))
    }

    /// Transpile the kernel function signature.
    fn transpile_kernel_signature(&self, func: &ItemFn) -> Result<String> {
        let name = func.sig.ident.to_string();

        let mut params = Vec::new();
        for param in &func.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                // Skip GridPos parameters
                if is_grid_pos_type(&pat_type.ty) {
                    continue;
                }

                let param_name = match pat_type.pat.as_ref() {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    _ => {
                        return Err(TranspileError::Unsupported(
                            "Complex pattern in parameter".into(),
                        ))
                    }
                };

                let cuda_type = self.type_mapper.map_type(&pat_type.ty)?;
                params.push(format!("{} {}", cuda_type.to_cuda_string(), param_name));
            }
        }

        Ok(format!("{}({})", name, params.join(", ")))
    }

    /// Transpile a block of statements.
    fn transpile_block(&mut self, block: &syn::Block) -> Result<String> {
        let mut output = String::new();

        for stmt in &block.stmts {
            let stmt_str = self.transpile_stmt(stmt)?;
            if !stmt_str.is_empty() {
                output.push_str(&stmt_str);
            }
        }

        Ok(output)
    }

    /// Transpile a single statement.
    fn transpile_stmt(&mut self, stmt: &Stmt) -> Result<String> {
        match stmt {
            Stmt::Local(local) => {
                let indent = self.indent_str();

                // Get variable name
                let var_name = match &local.pat {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    Pat::Type(pat_type) => {
                        if let Pat::Ident(ident) = pat_type.pat.as_ref() {
                            ident.ident.to_string()
                        } else {
                            return Err(TranspileError::Unsupported(
                                "Complex pattern in let binding".into(),
                            ));
                        }
                    }
                    _ => {
                        return Err(TranspileError::Unsupported(
                            "Complex pattern in let binding".into(),
                        ))
                    }
                };

                // Check for SharedTile or SharedArray type annotation
                if let Some(shared_decl) = self.try_parse_shared_declaration(local, &var_name)? {
                    // Register the shared variable
                    self.shared_vars.insert(var_name.clone(), SharedVarInfo {
                        name: var_name.clone(),
                        is_tile: shared_decl.dimensions.len() == 2,
                        dimensions: shared_decl.dimensions.clone(),
                        element_type: shared_decl.element_type.clone(),
                    });

                    // Add to shared memory config
                    self.shared_memory.add(shared_decl.clone());

                    // Return the __shared__ declaration
                    return Ok(format!("{indent}{}\n", shared_decl.to_cuda_decl()));
                }

                // Get initializer
                if let Some(init) = &local.init {
                    let expr_str = self.transpile_expr(&init.expr)?;

                    // Infer type from expression (default to float for now)
                    // In a more complete implementation, we'd do proper type inference
                    let type_str = self.infer_cuda_type(&init.expr);

                    Ok(format!("{indent}{type_str} {var_name} = {expr_str};\n"))
                } else {
                    // Uninitialized variable - need type annotation
                    Ok(format!("{indent}float {var_name};\n"))
                }
            }
            Stmt::Expr(expr, semi) => {
                let indent = self.indent_str();

                // Check if this is an if statement with early return (shouldn't wrap in return)
                if let Expr::If(if_expr) = expr {
                    // Check if the body contains only a return statement
                    if let Some(Stmt::Expr(Expr::Return(_), _)) = if_expr.then_branch.stmts.first() {
                        if if_expr.then_branch.stmts.len() == 1 && if_expr.else_branch.is_none() {
                            let expr_str = self.transpile_expr(expr)?;
                            return Ok(format!("{indent}{expr_str};\n"));
                        }
                    }
                }

                let expr_str = self.transpile_expr(expr)?;

                if semi.is_some() {
                    Ok(format!("{indent}{expr_str};\n"))
                } else {
                    // Expression without semicolon (implicit return)
                    // But check if it's already a return or an if with return
                    if matches!(expr, Expr::Return(_)) || expr_str.starts_with("return") || expr_str.starts_with("if (") {
                        Ok(format!("{indent}{expr_str};\n"))
                    } else {
                        Ok(format!("{indent}return {expr_str};\n"))
                    }
                }
            }
            Stmt::Item(_) => {
                // Items in function body not supported
                Err(TranspileError::Unsupported("Item in function body".into()))
            }
            Stmt::Macro(_) => Err(TranspileError::Unsupported("Macro in function body".into())),
        }
    }

    /// Transpile an expression.
    fn transpile_expr(&self, expr: &Expr) -> Result<String> {
        match expr {
            Expr::Lit(lit) => self.transpile_lit(lit),
            Expr::Path(path) => self.transpile_path(path),
            Expr::Binary(bin) => self.transpile_binary(bin),
            Expr::Unary(unary) => self.transpile_unary(unary),
            Expr::Paren(paren) => self.transpile_paren(paren),
            Expr::Index(index) => self.transpile_index(index),
            Expr::Call(call) => self.transpile_call(call),
            Expr::MethodCall(method) => self.transpile_method_call(method),
            Expr::If(if_expr) => self.transpile_if(if_expr),
            Expr::Assign(assign) => self.transpile_assign(assign),
            Expr::Cast(cast) => self.transpile_cast(cast),
            Expr::Match(match_expr) => self.transpile_match(match_expr),
            Expr::Block(block) => {
                // For block expressions, we just return the last expression
                if let Some(Stmt::Expr(expr, None)) = block.block.stmts.last() {
                    self.transpile_expr(expr)
                } else {
                    Err(TranspileError::Unsupported("Complex block expression".into()))
                }
            }
            Expr::Field(field) => {
                // Struct field access: obj.field -> obj.field
                let base = self.transpile_expr(&field.base)?;
                let member = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(idx) => idx.index.to_string(),
                };
                Ok(format!("{base}.{member}"))
            }
            Expr::Return(ret) => self.transpile_return(ret),
            Expr::ForLoop(for_loop) => self.transpile_for_loop(for_loop),
            Expr::While(while_loop) => self.transpile_while_loop(while_loop),
            Expr::Loop(loop_expr) => self.transpile_infinite_loop(loop_expr),
            Expr::Break(break_expr) => self.transpile_break(break_expr),
            Expr::Continue(cont_expr) => self.transpile_continue(cont_expr),
            Expr::Struct(struct_expr) => self.transpile_struct_literal(struct_expr),
            _ => Err(TranspileError::Unsupported(format!(
                "Expression type: {}",
                expr.to_token_stream()
            ))),
        }
    }

    /// Transpile a literal.
    fn transpile_lit(&self, lit: &ExprLit) -> Result<String> {
        match &lit.lit {
            Lit::Float(f) => {
                let s = f.to_string();
                // Ensure f32 literals have 'f' suffix in CUDA
                if s.ends_with("f32") || !s.contains('.') {
                    let num = s.trim_end_matches("f32").trim_end_matches("f64");
                    Ok(format!("{num}f"))
                } else if s.ends_with("f64") {
                    Ok(s.trim_end_matches("f64").to_string())
                } else {
                    // Plain float literal - add 'f' suffix for float
                    Ok(format!("{s}f"))
                }
            }
            Lit::Int(i) => Ok(i.to_string()),
            Lit::Bool(b) => Ok(if b.value { "1" } else { "0" }.to_string()),
            _ => Err(TranspileError::Unsupported(format!(
                "Literal type: {}",
                lit.to_token_stream()
            ))),
        }
    }

    /// Transpile a path (variable reference).
    fn transpile_path(&self, path: &ExprPath) -> Result<String> {
        let segments: Vec<_> = path
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect();

        if segments.len() == 1 {
            Ok(segments[0].clone())
        } else {
            Ok(segments.join("::"))
        }
    }

    /// Transpile a binary expression.
    fn transpile_binary(&self, bin: &ExprBinary) -> Result<String> {
        let left = self.transpile_expr(&bin.left)?;
        let right = self.transpile_expr(&bin.right)?;

        let op = match bin.op {
            BinOp::Add(_) => "+",
            BinOp::Sub(_) => "-",
            BinOp::Mul(_) => "*",
            BinOp::Div(_) => "/",
            BinOp::Rem(_) => "%",
            BinOp::And(_) => "&&",
            BinOp::Or(_) => "||",
            BinOp::BitXor(_) => "^",
            BinOp::BitAnd(_) => "&",
            BinOp::BitOr(_) => "|",
            BinOp::Shl(_) => "<<",
            BinOp::Shr(_) => ">>",
            BinOp::Eq(_) => "==",
            BinOp::Lt(_) => "<",
            BinOp::Le(_) => "<=",
            BinOp::Ne(_) => "!=",
            BinOp::Ge(_) => ">=",
            BinOp::Gt(_) => ">",
            BinOp::AddAssign(_) => "+=",
            BinOp::SubAssign(_) => "-=",
            BinOp::MulAssign(_) => "*=",
            BinOp::DivAssign(_) => "/=",
            BinOp::RemAssign(_) => "%=",
            BinOp::BitXorAssign(_) => "^=",
            BinOp::BitAndAssign(_) => "&=",
            BinOp::BitOrAssign(_) => "|=",
            BinOp::ShlAssign(_) => "<<=",
            BinOp::ShrAssign(_) => ">>=",
            _ => {
                return Err(TranspileError::Unsupported(format!(
                    "Binary operator: {}",
                    bin.to_token_stream()
                )))
            }
        };

        Ok(format!("{left} {op} {right}"))
    }

    /// Transpile a unary expression.
    fn transpile_unary(&self, unary: &ExprUnary) -> Result<String> {
        let expr = self.transpile_expr(&unary.expr)?;

        let op = match unary.op {
            UnOp::Neg(_) => "-",
            UnOp::Not(_) => "!",
            UnOp::Deref(_) => "*",
            _ => {
                return Err(TranspileError::Unsupported(format!(
                    "Unary operator: {}",
                    unary.to_token_stream()
                )))
            }
        };

        Ok(format!("{op}({expr})"))
    }

    /// Transpile a parenthesized expression.
    fn transpile_paren(&self, paren: &ExprParen) -> Result<String> {
        let inner = self.transpile_expr(&paren.expr)?;
        Ok(format!("({inner})"))
    }

    /// Transpile an index expression.
    fn transpile_index(&self, index: &ExprIndex) -> Result<String> {
        let base = self.transpile_expr(&index.expr)?;
        let idx = self.transpile_expr(&index.index)?;
        Ok(format!("{base}[{idx}]"))
    }

    /// Transpile a function call.
    fn transpile_call(&self, call: &ExprCall) -> Result<String> {
        let func = self.transpile_expr(&call.func)?;

        // Check for intrinsics
        if let Some(intrinsic) = self.intrinsics.lookup(&func) {
            let cuda_name = intrinsic.to_cuda_string();

            // Check if this is a "value" intrinsic (like threadIdx.x) vs a "function" intrinsic
            // Value intrinsics don't have parentheses in CUDA
            let is_value_intrinsic = cuda_name.contains("Idx.")
                || cuda_name.contains("Dim.")
                || cuda_name.starts_with("threadIdx")
                || cuda_name.starts_with("blockIdx")
                || cuda_name.starts_with("blockDim")
                || cuda_name.starts_with("gridDim");

            if is_value_intrinsic && call.args.is_empty() {
                // Value intrinsics: threadIdx.x, blockIdx.y, etc. - no parens
                return Ok(cuda_name.to_string());
            }

            if call.args.is_empty() && cuda_name.ends_with("()") {
                // Zero-arg function intrinsics: __syncthreads()
                return Ok(cuda_name.to_string());
            }

            let args: Vec<String> = call
                .args
                .iter()
                .map(|a| self.transpile_expr(a))
                .collect::<Result<_>>()?;

            return Ok(format!("{}({})", cuda_name.trim_end_matches("()"), args.join(", ")));
        }

        // Regular function call
        let args: Vec<String> = call
            .args
            .iter()
            .map(|a| self.transpile_expr(a))
            .collect::<Result<_>>()?;

        Ok(format!("{}({})", func, args.join(", ")))
    }

    /// Transpile a method call, handling stencil intrinsics.
    fn transpile_method_call(&self, method: &ExprMethodCall) -> Result<String> {
        let receiver = self.transpile_expr(&method.receiver)?;
        let method_name = method.method.to_string();

        // Check if this is a SharedTile/SharedArray method call
        if let Some(result) = self.try_transpile_shared_method_call(&receiver, &method_name, &method.args) {
            return result;
        }

        // Check if this is a RingContext method call (in ring kernel mode)
        if self.ring_kernel_mode && self.context_vars.contains(&receiver) {
            return self.transpile_context_method(&method_name, &method.args);
        }

        // Check if this is a GridPos method call
        if self.grid_pos_vars.contains(&receiver) {
            return self.transpile_stencil_intrinsic(&method_name, &method.args);
        }

        // Check for ring kernel intrinsics (like is_active(), should_terminate())
        if self.ring_kernel_mode {
            if let Some(intrinsic) = RingKernelIntrinsic::from_name(&method_name) {
                let args: Vec<String> = method.args.iter()
                    .map(|a| self.transpile_expr(a).unwrap_or_default())
                    .collect();
                return Ok(intrinsic.to_cuda(&args));
            }
        }

        // Check for f32/f64 math methods
        if let Some(intrinsic) = self.intrinsics.lookup(&method_name) {
            let cuda_name = intrinsic.to_cuda_string();
            let args: Vec<String> = std::iter::once(receiver)
                .chain(method.args.iter().map(|a| self.transpile_expr(a).unwrap_or_default()))
                .collect();

            return Ok(format!("{}({})", cuda_name, args.join(", ")));
        }

        // Regular method call (treat as field access + call for C structs)
        let args: Vec<String> = method
            .args
            .iter()
            .map(|a| self.transpile_expr(a))
            .collect::<Result<_>>()?;

        Ok(format!("{}.{}({})", receiver, method_name, args.join(", ")))
    }

    /// Transpile a RingContext method call to CUDA intrinsics.
    fn transpile_context_method(
        &self,
        method: &str,
        args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    ) -> Result<String> {
        let ctx_method = ContextMethod::from_name(method).ok_or_else(|| {
            TranspileError::Unsupported(format!("Unknown context method: {}", method))
        })?;

        let cuda_args: Vec<String> = args.iter()
            .map(|a| self.transpile_expr(a).unwrap_or_default())
            .collect();

        Ok(ctx_method.to_cuda(&cuda_args))
    }

    /// Transpile a stencil intrinsic method call.
    fn transpile_stencil_intrinsic(
        &self,
        method: &str,
        args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    ) -> Result<String> {
        let config = self.config.as_ref().ok_or_else(|| {
            TranspileError::Unsupported("Stencil intrinsic without config".into())
        })?;

        let buffer_width = config.buffer_width().to_string();

        let intrinsic =
            StencilIntrinsic::from_method_name(method).ok_or_else(|| {
                TranspileError::Unsupported(format!("Unknown stencil intrinsic: {method}"))
            })?;

        match intrinsic {
            StencilIntrinsic::Index => {
                // pos.idx() -> idx
                Ok("idx".to_string())
            }
            StencilIntrinsic::North
            | StencilIntrinsic::South
            | StencilIntrinsic::East
            | StencilIntrinsic::West => {
                // pos.north(buf) -> buf[idx - buffer_width]
                if args.is_empty() {
                    return Err(TranspileError::Unsupported(
                        "Stencil accessor requires buffer argument".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                Ok(intrinsic.to_cuda_index_2d(&buffer, &buffer_width, "idx"))
            }
            StencilIntrinsic::At => {
                // pos.at(buf, dx, dy) -> buf[idx + dy * buffer_width + dx]
                if args.len() < 3 {
                    return Err(TranspileError::Unsupported(
                        "at() requires buffer, dx, dy arguments".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                let dx = self.transpile_expr(&args[1])?;
                let dy = self.transpile_expr(&args[2])?;
                Ok(format!(
                    "{buffer}[idx + ({dy}) * {buffer_width} + ({dx})]"
                ))
            }
            StencilIntrinsic::Up | StencilIntrinsic::Down => {
                // 3D intrinsics
                Err(TranspileError::Unsupported(
                    "3D stencil intrinsics not yet implemented".into(),
                ))
            }
        }
    }

    /// Transpile an if expression.
    fn transpile_if(&self, if_expr: &ExprIf) -> Result<String> {
        let cond = self.transpile_expr(&if_expr.cond)?;

        // Check if the body contains only a return statement (early return pattern)
        if let Some(Stmt::Expr(Expr::Return(ret), _)) = if_expr.then_branch.stmts.first() {
            if if_expr.then_branch.stmts.len() == 1 && if_expr.else_branch.is_none() {
                // Simple early return: if (cond) return;
                if ret.expr.is_none() {
                    return Ok(format!("if ({cond}) return"));
                }
                let ret_val = self.transpile_expr(ret.expr.as_ref().unwrap())?;
                return Ok(format!("if ({cond}) return {ret_val}"));
            }
        }

        // For now, only handle if-else as ternary when it's an expression
        if let Some((_, else_branch)) = &if_expr.else_branch {
            // If both branches are simple expressions, use ternary
            if let (Some(Stmt::Expr(then_expr, None)), Expr::Block(else_block)) =
                (if_expr.then_branch.stmts.last(), else_branch.as_ref())
            {
                if let Some(Stmt::Expr(else_expr, None)) = else_block.block.stmts.last() {
                    let then_str = self.transpile_expr(then_expr)?;
                    let else_str = self.transpile_expr(else_expr)?;
                    return Ok(format!("({cond}) ? ({then_str}) : ({else_str})"));
                }
            }

            // Otherwise, generate if statement
            if let Expr::If(else_if) = else_branch.as_ref() {
                // else if chain
                let then_body = self.transpile_if_body(&if_expr.then_branch)?;
                let else_part = self.transpile_if(else_if)?;
                return Ok(format!("if ({cond}) {{{then_body}}} else {else_part}"));
            } else if let Expr::Block(else_block) = else_branch.as_ref() {
                // else block
                let then_body = self.transpile_if_body(&if_expr.then_branch)?;
                let else_body = self.transpile_if_body(&else_block.block)?;
                return Ok(format!("if ({cond}) {{{then_body}}} else {{{else_body}}}"));
            }
        }

        // If without else
        let then_body = self.transpile_if_body(&if_expr.then_branch)?;
        Ok(format!("if ({cond}) {{{then_body}}}"))
    }

    /// Transpile the body of an if branch.
    fn transpile_if_body(&self, block: &syn::Block) -> Result<String> {
        let mut body = String::new();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Expr(expr, Some(_)) => {
                    let expr_str = self.transpile_expr(expr)?;
                    body.push_str(&format!(" {expr_str};"));
                }
                Stmt::Expr(Expr::Return(ret), None) => {
                    // Handle explicit return without semicolon
                    if let Some(ret_expr) = &ret.expr {
                        let expr_str = self.transpile_expr(ret_expr)?;
                        body.push_str(&format!(" return {expr_str};"));
                    } else {
                        body.push_str(" return;");
                    }
                }
                Stmt::Expr(expr, None) => {
                    let expr_str = self.transpile_expr(expr)?;
                    body.push_str(&format!(" return {expr_str};"));
                }
                _ => {}
            }
        }
        Ok(body)
    }

    /// Transpile an assignment expression.
    fn transpile_assign(&self, assign: &ExprAssign) -> Result<String> {
        let left = self.transpile_expr(&assign.left)?;
        let right = self.transpile_expr(&assign.right)?;
        Ok(format!("{left} = {right}"))
    }

    /// Transpile a cast expression.
    fn transpile_cast(&self, cast: &ExprCast) -> Result<String> {
        let expr = self.transpile_expr(&cast.expr)?;
        let cuda_type = self.type_mapper.map_type(&cast.ty)?;
        Ok(format!("({})({})", cuda_type.to_cuda_string(), expr))
    }

    /// Transpile a return expression.
    fn transpile_return(&self, ret: &ExprReturn) -> Result<String> {
        if let Some(expr) = &ret.expr {
            let expr_str = self.transpile_expr(expr)?;
            Ok(format!("return {expr_str}"))
        } else {
            Ok("return".to_string())
        }
    }

    /// Transpile a struct literal expression.
    ///
    /// Converts Rust struct literals to C-style compound literals:
    /// `Point { x: 1.0, y: 2.0 }` -> `(Point){ .x = 1.0f, .y = 2.0f }`
    fn transpile_struct_literal(&self, struct_expr: &ExprStruct) -> Result<String> {
        // Get the struct type name
        let type_name = struct_expr
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");

        // Transpile each field
        let mut fields = Vec::new();
        for field in &struct_expr.fields {
            let field_name = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(idx) => idx.index.to_string(),
            };
            let value = self.transpile_expr(&field.expr)?;
            fields.push(format!(".{} = {}", field_name, value));
        }

        // Check for struct update syntax (not supported in C)
        if struct_expr.rest.is_some() {
            return Err(TranspileError::Unsupported(
                "Struct update syntax (..base) is not supported in CUDA".into(),
            ));
        }

        // Generate C compound literal: (TypeName){ .field1 = val1, .field2 = val2 }
        Ok(format!("({}){{ {} }}", type_name, fields.join(", ")))
    }

    // === Loop Transpilation ===

    /// Transpile a for loop to CUDA.
    ///
    /// Handles `for i in start..end` and `for i in start..=end` patterns.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Rust
    /// for i in 0..n {
    ///     data[i] = 0.0;
    /// }
    ///
    /// // CUDA
    /// for (int i = 0; i < n; i++) {
    ///     data[i] = 0.0f;
    /// }
    /// ```
    fn transpile_for_loop(&self, for_loop: &ExprForLoop) -> Result<String> {
        // Check if loops are allowed
        if !self.validation_mode.allows_loops() {
            return Err(TranspileError::Unsupported(
                "Loops are not allowed in stencil kernels".into(),
            ));
        }

        // Extract loop variable name
        let var_name = extract_loop_var(&for_loop.pat).ok_or_else(|| {
            TranspileError::Unsupported("Complex pattern in for loop".into())
        })?;

        // The iterator expression should be a range
        let header = match for_loop.expr.as_ref() {
            Expr::Range(range) => {
                let range_info = RangeInfo::from_range(range, |e| self.transpile_expr(e));
                range_info.to_cuda_for_header(&var_name, "int")
            }
            _ => {
                // For non-range iterators, we can't directly transpile
                return Err(TranspileError::Unsupported(
                    "Only range expressions (start..end) are supported in for loops".into(),
                ));
            }
        };

        // Transpile the loop body
        let body = self.transpile_loop_body(&for_loop.body)?;

        Ok(format!("{header} {{\n{body}}}"))
    }

    /// Transpile a while loop to CUDA.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Rust
    /// while !done {
    ///     process();
    /// }
    ///
    /// // CUDA
    /// while (!done) {
    ///     process();
    /// }
    /// ```
    fn transpile_while_loop(&self, while_loop: &ExprWhile) -> Result<String> {
        // Check if loops are allowed
        if !self.validation_mode.allows_loops() {
            return Err(TranspileError::Unsupported(
                "Loops are not allowed in stencil kernels".into(),
            ));
        }

        // Transpile the condition
        let condition = self.transpile_expr(&while_loop.cond)?;

        // Transpile the loop body
        let body = self.transpile_loop_body(&while_loop.body)?;

        Ok(format!("while ({condition}) {{\n{body}}}"))
    }

    /// Transpile an infinite loop to CUDA.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Rust
    /// loop {
    ///     if should_exit { break; }
    /// }
    ///
    /// // CUDA
    /// while (true) {
    ///     if (should_exit) { break; }
    /// }
    /// ```
    fn transpile_infinite_loop(&self, loop_expr: &ExprLoop) -> Result<String> {
        // Check if loops are allowed
        if !self.validation_mode.allows_loops() {
            return Err(TranspileError::Unsupported(
                "Loops are not allowed in stencil kernels".into(),
            ));
        }

        // Transpile the loop body
        let body = self.transpile_loop_body(&loop_expr.body)?;

        // Use while(true) for infinite loops
        Ok(format!("while (true) {{\n{body}}}"))
    }

    /// Transpile a break expression.
    fn transpile_break(&self, break_expr: &ExprBreak) -> Result<String> {
        // Check for labeled break (not supported)
        if break_expr.label.is_some() {
            return Err(TranspileError::Unsupported(
                "Labeled break is not supported in CUDA".into(),
            ));
        }

        // Check for break with value (not supported in CUDA)
        if break_expr.expr.is_some() {
            return Err(TranspileError::Unsupported(
                "Break with value is not supported in CUDA".into(),
            ));
        }

        Ok("break".to_string())
    }

    /// Transpile a continue expression.
    fn transpile_continue(&self, cont_expr: &ExprContinue) -> Result<String> {
        // Check for labeled continue (not supported)
        if cont_expr.label.is_some() {
            return Err(TranspileError::Unsupported(
                "Labeled continue is not supported in CUDA".into(),
            ));
        }

        Ok("continue".to_string())
    }

    /// Transpile a loop body (block of statements).
    fn transpile_loop_body(&self, block: &syn::Block) -> Result<String> {
        let mut output = String::new();
        let inner_indent = "    ".repeat(self.indent + 1);

        for stmt in &block.stmts {
            match stmt {
                Stmt::Local(local) => {
                    // Variable declaration
                    let var_name = match &local.pat {
                        Pat::Ident(ident) => ident.ident.to_string(),
                        Pat::Type(pat_type) => {
                            if let Pat::Ident(ident) = pat_type.pat.as_ref() {
                                ident.ident.to_string()
                            } else {
                                return Err(TranspileError::Unsupported(
                                    "Complex pattern in let binding".into(),
                                ));
                            }
                        }
                        _ => {
                            return Err(TranspileError::Unsupported(
                                "Complex pattern in let binding".into(),
                            ))
                        }
                    };

                    if let Some(init) = &local.init {
                        let expr_str = self.transpile_expr(&init.expr)?;
                        let type_str = self.infer_cuda_type(&init.expr);
                        output.push_str(&format!("{inner_indent}{type_str} {var_name} = {expr_str};\n"));
                    } else {
                        output.push_str(&format!("{inner_indent}float {var_name};\n"));
                    }
                }
                Stmt::Expr(expr, semi) => {
                    let expr_str = self.transpile_expr(expr)?;
                    if semi.is_some() {
                        output.push_str(&format!("{inner_indent}{expr_str};\n"));
                    } else {
                        // Expression without semicolon at end of block
                        output.push_str(&format!("{inner_indent}{expr_str};\n"));
                    }
                }
                _ => {
                    return Err(TranspileError::Unsupported(
                        "Unsupported statement in loop body".into(),
                    ));
                }
            }
        }

        // Add closing indentation
        let closing_indent = "    ".repeat(self.indent);
        output.push_str(&closing_indent);

        Ok(output)
    }

    // === Shared Memory Support ===

    /// Try to parse a local variable declaration as a shared memory declaration.
    ///
    /// Recognizes patterns like:
    /// - `let tile = SharedTile::<f32, 16, 16>::new();`
    /// - `let buffer = SharedArray::<f32, 256>::new();`
    /// - `let tile: SharedTile<f32, 16, 16> = SharedTile::new();`
    fn try_parse_shared_declaration(
        &self,
        local: &syn::Local,
        var_name: &str,
    ) -> Result<Option<SharedMemoryDecl>> {
        // Check if there's a type annotation
        if let Pat::Type(pat_type) = &local.pat {
            let type_str = pat_type.ty.to_token_stream().to_string();
            return self.parse_shared_type(&type_str, var_name);
        }

        // Check the initializer expression for SharedTile::new() or SharedArray::new()
        if let Some(init) = &local.init {
            if let Expr::Call(call) = init.expr.as_ref() {
                if let Expr::Path(path) = call.func.as_ref() {
                    let path_str = path.to_token_stream().to_string();
                    return self.parse_shared_type(&path_str, var_name);
                }
            }
        }

        Ok(None)
    }

    /// Parse a type string to extract shared memory info.
    fn parse_shared_type(&self, type_str: &str, var_name: &str) -> Result<Option<SharedMemoryDecl>> {
        // Clean up the type string (remove spaces around ::)
        let type_str = type_str.replace(" :: ", "::").replace(" ::", "::").replace(":: ", "::");

        // Check for SharedTile<T, W, H> or SharedTile::<T, W, H>::new
        if type_str.contains("SharedTile") {
            // Extract the generic parameters
            if let Some(start) = type_str.find('<') {
                if let Some(end) = type_str.rfind('>') {
                    let params = &type_str[start + 1..end];
                    let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

                    if parts.len() >= 3 {
                        let rust_type = parts[0];
                        let width: usize = parts[1].parse().map_err(|_| {
                            TranspileError::Unsupported("Invalid SharedTile width".into())
                        })?;
                        let height: usize = parts[2].parse().map_err(|_| {
                            TranspileError::Unsupported("Invalid SharedTile height".into())
                        })?;

                        let cuda_type = rust_to_cuda_element_type(rust_type);
                        return Ok(Some(SharedMemoryDecl::tile(
                            var_name,
                            cuda_type,
                            width,
                            height,
                        )));
                    }
                }
            }
        }

        // Check for SharedArray<T, N> or SharedArray::<T, N>::new
        if type_str.contains("SharedArray") {
            if let Some(start) = type_str.find('<') {
                if let Some(end) = type_str.rfind('>') {
                    let params = &type_str[start + 1..end];
                    let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

                    if parts.len() >= 2 {
                        let rust_type = parts[0];
                        let size: usize = parts[1].parse().map_err(|_| {
                            TranspileError::Unsupported("Invalid SharedArray size".into())
                        })?;

                        let cuda_type = rust_to_cuda_element_type(rust_type);
                        return Ok(Some(SharedMemoryDecl::array(var_name, cuda_type, size)));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Check if a variable is a shared memory variable and handle method calls.
    fn try_transpile_shared_method_call(
        &self,
        receiver: &str,
        method_name: &str,
        args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    ) -> Option<Result<String>> {
        let shared_info = self.shared_vars.get(receiver)?;

        match method_name {
            "get" => {
                // tile.get(x, y) -> tile[y][x] (for 2D) or arr[idx] (for 1D)
                if shared_info.is_tile {
                    if args.len() >= 2 {
                        let x = self.transpile_expr(&args[0]).ok()?;
                        let y = self.transpile_expr(&args[1]).ok()?;
                        // CUDA uses row-major: tile[row][col] = tile[y][x]
                        Some(Ok(format!("{}[{}][{}]", receiver, y, x)))
                    } else {
                        Some(Err(TranspileError::Unsupported(
                            "SharedTile.get requires x and y arguments".into(),
                        )))
                    }
                } else {
                    // 1D array
                    if args.len() >= 1 {
                        let idx = self.transpile_expr(&args[0]).ok()?;
                        Some(Ok(format!("{}[{}]", receiver, idx)))
                    } else {
                        Some(Err(TranspileError::Unsupported(
                            "SharedArray.get requires index argument".into(),
                        )))
                    }
                }
            }
            "set" => {
                // tile.set(x, y, val) -> tile[y][x] = val
                if shared_info.is_tile {
                    if args.len() >= 3 {
                        let x = self.transpile_expr(&args[0]).ok()?;
                        let y = self.transpile_expr(&args[1]).ok()?;
                        let val = self.transpile_expr(&args[2]).ok()?;
                        Some(Ok(format!("{}[{}][{}] = {}", receiver, y, x, val)))
                    } else {
                        Some(Err(TranspileError::Unsupported(
                            "SharedTile.set requires x, y, and value arguments".into(),
                        )))
                    }
                } else {
                    // 1D array
                    if args.len() >= 2 {
                        let idx = self.transpile_expr(&args[0]).ok()?;
                        let val = self.transpile_expr(&args[1]).ok()?;
                        Some(Ok(format!("{}[{}] = {}", receiver, idx, val)))
                    } else {
                        Some(Err(TranspileError::Unsupported(
                            "SharedArray.set requires index and value arguments".into(),
                        )))
                    }
                }
            }
            "width" | "height" | "size" => {
                // These are compile-time constants
                match method_name {
                    "width" if shared_info.is_tile => {
                        Some(Ok(shared_info.dimensions[1].to_string()))
                    }
                    "height" if shared_info.is_tile => {
                        Some(Ok(shared_info.dimensions[0].to_string()))
                    }
                    "size" => {
                        let total: usize = shared_info.dimensions.iter().product();
                        Some(Ok(total.to_string()))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Transpile a match expression to switch/case.
    fn transpile_match(&self, match_expr: &ExprMatch) -> Result<String> {
        let scrutinee = self.transpile_expr(&match_expr.expr)?;
        let mut output = format!("switch ({scrutinee}) {{\n");

        for arm in &match_expr.arms {
            // Handle the pattern
            let case_label = self.transpile_match_pattern(&arm.pat)?;

            if case_label == "default" || case_label.starts_with("/*") {
                output.push_str("        default: {\n");
            } else {
                output.push_str(&format!("        case {case_label}: {{\n"));
            }

            // Handle the arm body - check if it's a block with statements
            match arm.body.as_ref() {
                Expr::Block(block) => {
                    // Block expression with multiple statements
                    for stmt in &block.block.stmts {
                        let stmt_str = self.transpile_stmt_inline(stmt)?;
                        output.push_str(&format!("            {stmt_str}\n"));
                    }
                }
                _ => {
                    // Single expression - wrap in statement
                    let body = self.transpile_expr(&arm.body)?;
                    output.push_str(&format!("            {body};\n"));
                }
            }

            output.push_str("            break;\n");
            output.push_str("        }\n");
        }

        output.push_str("    }");
        Ok(output)
    }

    /// Transpile a match pattern to a case label.
    fn transpile_match_pattern(&self, pat: &Pat) -> Result<String> {
        match pat {
            Pat::Lit(pat_lit) => {
                // Integer literal pattern - pat_lit.lit contains the literal
                match &pat_lit.lit {
                    Lit::Int(i) => Ok(i.to_string()),
                    Lit::Bool(b) => Ok(if b.value { "1" } else { "0" }.to_string()),
                    _ => Err(TranspileError::Unsupported(
                        "Non-integer literal in match pattern".into(),
                    )),
                }
            }
            Pat::Wild(_) => {
                // _ pattern becomes default
                Ok("default".to_string())
            }
            Pat::Ident(ident) => {
                // Named pattern - treat as default case for now
                // This handles things like `x => ...` which bind a value
                Ok(format!("/* {} */ default", ident.ident))
            }
            Pat::Or(pat_or) => {
                // Multiple patterns: 0 | 1 | 2 => ...
                // CUDA switch doesn't support this directly, we need multiple case labels
                // For now, just use the first pattern and note the limitation
                if let Some(first) = pat_or.cases.first() {
                    self.transpile_match_pattern(first)
                } else {
                    Err(TranspileError::Unsupported("Empty or pattern".into()))
                }
            }
            _ => Err(TranspileError::Unsupported(format!(
                "Match pattern: {}",
                pat.to_token_stream()
            ))),
        }
    }

    /// Transpile a statement without indentation (for inline use in switch).
    fn transpile_stmt_inline(&self, stmt: &Stmt) -> Result<String> {
        match stmt {
            Stmt::Local(local) => {
                let var_name = match &local.pat {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    Pat::Type(pat_type) => {
                        if let Pat::Ident(ident) = pat_type.pat.as_ref() {
                            ident.ident.to_string()
                        } else {
                            return Err(TranspileError::Unsupported(
                                "Complex pattern in let binding".into(),
                            ));
                        }
                    }
                    _ => {
                        return Err(TranspileError::Unsupported(
                            "Complex pattern in let binding".into(),
                        ))
                    }
                };

                if let Some(init) = &local.init {
                    let expr_str = self.transpile_expr(&init.expr)?;
                    let type_str = self.infer_cuda_type(&init.expr);
                    Ok(format!("{type_str} {var_name} = {expr_str};"))
                } else {
                    Ok(format!("float {var_name};"))
                }
            }
            Stmt::Expr(expr, semi) => {
                let expr_str = self.transpile_expr(expr)?;
                if semi.is_some() {
                    Ok(format!("{expr_str};"))
                } else {
                    Ok(format!("return {expr_str};"))
                }
            }
            _ => Err(TranspileError::Unsupported("Unsupported statement in match arm".into())),
        }
    }

    /// Infer CUDA type from expression (simple heuristic).
    fn infer_cuda_type(&self, expr: &Expr) -> &'static str {
        match expr {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Float(_) => "float",
                Lit::Int(_) => "int",
                Lit::Bool(_) => "int",
                _ => "float",
            },
            Expr::Binary(bin) => {
                // Check if this is an integer operation
                let left_type = self.infer_cuda_type(&bin.left);
                let right_type = self.infer_cuda_type(&bin.right);
                // If both sides are int, result is int
                if left_type == "int" && right_type == "int" {
                    "int"
                } else {
                    "float"
                }
            }
            Expr::Call(call) => {
                // Check for intrinsics that return int
                if let Ok(func) = self.transpile_expr(&call.func) {
                    if let Some(intrinsic) = self.intrinsics.lookup(&func) {
                        let cuda_name = intrinsic.to_cuda_string();
                        // Thread/block indices return int
                        if cuda_name.contains("Idx") || cuda_name.contains("Dim") {
                            return "int";
                        }
                    }
                }
                "float"
            }
            Expr::Index(_) => "float", // Array access - could be any type, default to float
            Expr::Cast(cast) => {
                // Use the target type of the cast
                if let Ok(cuda_type) = self.type_mapper.map_type(&cast.ty) {
                    let s = cuda_type.to_cuda_string();
                    if s.contains("int") || s.contains("size_t") || s == "unsigned long long" {
                        return "int";
                    }
                }
                "float"
            }
            Expr::MethodCall(_) => "float",
            _ => "float",
        }
    }
}

/// Transpile a function to CUDA without stencil configuration.
pub fn transpile_function(func: &ItemFn) -> Result<String> {
    let mut transpiler = CudaTranspiler::new_generic();

    // Generate function signature
    let name = func.sig.ident.to_string();

    let mut params = Vec::new();
    for param in &func.sig.inputs {
        if let FnArg::Typed(pat_type) = param {
            let param_name = match pat_type.pat.as_ref() {
                Pat::Ident(ident) => ident.ident.to_string(),
                _ => continue,
            };

            let cuda_type = transpiler.type_mapper.map_type(&pat_type.ty)?;
            params.push(format!("{} {}", cuda_type.to_cuda_string(), param_name));
        }
    }

    // Return type
    let return_type = match &func.sig.output {
        ReturnType::Default => "void".to_string(),
        ReturnType::Type(_, ty) => transpiler.type_mapper.map_type(ty)?.to_cuda_string(),
    };

    // Generate body
    let body = transpiler.transpile_block(&func.block)?;

    Ok(format!(
        "__device__ {return_type} {name}({params}) {{\n{body}}}\n",
        params = params.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_simple_arithmetic() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote!(a + b * 2.0);
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "a + b * 2.0f");
    }

    #[test]
    fn test_let_binding() {
        let mut transpiler = CudaTranspiler::new_generic();

        let stmt: Stmt = parse_quote!(let x = a + b;);
        let result = transpiler.transpile_stmt(&stmt).unwrap();
        assert!(result.contains("float x = a + b;"));
    }

    #[test]
    fn test_array_index() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote!(data[idx]);
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "data[idx]");
    }

    #[test]
    fn test_stencil_intrinsics() {
        let config = StencilConfig::new("test")
            .with_tile_size(16, 16)
            .with_halo(1);
        let mut transpiler = CudaTranspiler::new(config);
        transpiler.grid_pos_vars.push("pos".to_string());

        // Test pos.idx()
        let expr: Expr = parse_quote!(pos.idx());
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "idx");

        // Test pos.north(p)
        let expr: Expr = parse_quote!(pos.north(p));
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "p[idx - 18]");

        // Test pos.east(p)
        let expr: Expr = parse_quote!(pos.east(p));
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "p[idx + 1]");
    }

    #[test]
    fn test_ternary_if() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote!(if x > 0.0 { x } else { -x });
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("?"));
        assert!(result.contains(":"));
    }

    #[test]
    fn test_full_stencil_kernel() {
        let func: ItemFn = parse_quote! {
            fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
                let curr = p[pos.idx()];
                let prev = p_prev[pos.idx()];
                let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
                p_prev[pos.idx()] = (2.0 * curr - prev + c2 * lap);
            }
        };

        let config = StencilConfig::new("fdtd")
            .with_tile_size(16, 16)
            .with_halo(1);

        let mut transpiler = CudaTranspiler::new(config);
        let cuda = transpiler.transpile_stencil(&func).unwrap();

        // Check key features
        assert!(cuda.contains("extern \"C\" __global__"));
        assert!(cuda.contains("threadIdx.x"));
        assert!(cuda.contains("threadIdx.y"));
        assert!(cuda.contains("buffer_width = 18"));
        assert!(cuda.contains("const float* __restrict__ p"));
        assert!(cuda.contains("float* __restrict__ p_prev"));
        assert!(!cuda.contains("GridPos")); // GridPos should be removed

        println!("Generated CUDA:\n{}", cuda);
    }

    #[test]
    fn test_early_return() {
        let mut transpiler = CudaTranspiler::new_generic();

        let stmt: Stmt = parse_quote!(return;);
        let result = transpiler.transpile_stmt(&stmt).unwrap();
        assert!(result.contains("return;"));

        let stmt_val: Stmt = parse_quote!(return 42;);
        let result_val = transpiler.transpile_stmt(&stmt_val).unwrap();
        assert!(result_val.contains("return 42;"));
    }

    #[test]
    fn test_match_to_switch() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            match edge {
                0 => { idx = 1 * 18 + i; }
                1 => { idx = 16 * 18 + i; }
                _ => { idx = 0; }
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("switch (edge)"), "Should generate switch: {}", result);
        assert!(result.contains("case 0:"), "Should have case 0: {}", result);
        assert!(result.contains("case 1:"), "Should have case 1: {}", result);
        assert!(result.contains("default:"), "Should have default: {}", result);
        assert!(result.contains("break;"), "Should have break: {}", result);

        println!("Generated switch:\n{}", result);
    }

    #[test]
    fn test_block_idx_intrinsics() {
        let transpiler = CudaTranspiler::new_generic();

        // Test block_idx_x() call
        let expr: Expr = parse_quote!(block_idx_x());
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "blockIdx.x");

        // Test thread_idx_y() call
        let expr2: Expr = parse_quote!(thread_idx_y());
        let result2 = transpiler.transpile_expr(&expr2).unwrap();
        assert_eq!(result2, "threadIdx.y");

        // Test grid_dim_x() call
        let expr3: Expr = parse_quote!(grid_dim_x());
        let result3 = transpiler.transpile_expr(&expr3).unwrap();
        assert_eq!(result3, "gridDim.x");
    }

    #[test]
    fn test_global_index_calculation() {
        let transpiler = CudaTranspiler::new_generic();

        // Common CUDA pattern: gx = blockIdx.x * blockDim.x + threadIdx.x
        let expr: Expr = parse_quote!(block_idx_x() * block_dim_x() + thread_idx_x());
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("blockIdx.x"), "Should contain blockIdx.x");
        assert!(result.contains("blockDim.x"), "Should contain blockDim.x");
        assert!(result.contains("threadIdx.x"), "Should contain threadIdx.x");

        println!("Global index expression: {}", result);
    }

    // === Loop Transpilation Tests ===

    #[test]
    fn test_for_loop_transpile() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            for i in 0..n {
                data[i] = 0.0;
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("for (int i = 0; i < n; i++)"), "Should generate for loop header: {}", result);
        assert!(result.contains("data[i] = 0.0f"), "Should contain loop body: {}", result);

        println!("Generated for loop:\n{}", result);
    }

    #[test]
    fn test_for_loop_inclusive_range() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            for i in 1..=10 {
                sum += i;
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("for (int i = 1; i <= 10; i++)"), "Should generate inclusive range: {}", result);

        println!("Generated inclusive for loop:\n{}", result);
    }

    #[test]
    fn test_while_loop_transpile() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            while i < 10 {
                i += 1;
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("while (i < 10)"), "Should generate while loop: {}", result);
        assert!(result.contains("i += 1"), "Should contain loop body: {}", result);

        println!("Generated while loop:\n{}", result);
    }

    #[test]
    fn test_while_loop_negation() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            while !done {
                process();
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("while (!(done))"), "Should negate condition: {}", result);

        println!("Generated while loop with negation:\n{}", result);
    }

    #[test]
    fn test_infinite_loop_transpile() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            loop {
                process();
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("while (true)"), "Should generate infinite loop: {}", result);
        assert!(result.contains("process()"), "Should contain loop body: {}", result);

        println!("Generated infinite loop:\n{}", result);
    }

    #[test]
    fn test_break_transpile() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote!(break);
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "break");
    }

    #[test]
    fn test_continue_transpile() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote!(continue);
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "continue");
    }

    #[test]
    fn test_loop_with_break() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            loop {
                if done {
                    break;
                }
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("while (true)"), "Should generate infinite loop: {}", result);
        assert!(result.contains("break"), "Should contain break: {}", result);

        println!("Generated loop with break:\n{}", result);
    }

    #[test]
    fn test_nested_loops() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            for i in 0..m {
                for j in 0..n {
                    matrix[i * n + j] = 0.0;
                }
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("for (int i = 0; i < m; i++)"), "Should have outer loop: {}", result);
        assert!(result.contains("for (int j = 0; j < n; j++)"), "Should have inner loop: {}", result);

        println!("Generated nested loops:\n{}", result);
    }

    #[test]
    fn test_stencil_mode_rejects_loops() {
        let config = StencilConfig::new("test").with_tile_size(16, 16).with_halo(1);
        let transpiler = CudaTranspiler::new(config);

        let expr: Expr = parse_quote! {
            for i in 0..n {
                data[i] = 0.0;
            }
        };

        let result = transpiler.transpile_expr(&expr);
        assert!(result.is_err(), "Stencil mode should reject loops");
    }

    #[test]
    fn test_labeled_break_rejected() {
        let transpiler = CudaTranspiler::new_generic();

        // Note: We can't directly parse `break 'label` without a labeled block,
        // so we test that the error path exists by checking the function handles labels
        let break_expr = syn::ExprBreak {
            attrs: Vec::new(),
            break_token: syn::token::Break::default(),
            label: Some(syn::Lifetime::new("'outer", proc_macro2::Span::call_site())),
            expr: None,
        };

        let result = transpiler.transpile_break(&break_expr);
        assert!(result.is_err(), "Labeled break should be rejected");
    }

    #[test]
    fn test_full_kernel_with_loop() {
        let func: ItemFn = parse_quote! {
            fn fill_array(data: &mut [f32], n: i32) {
                for i in 0..n {
                    data[i as usize] = 0.0;
                }
            }
        };

        let mut transpiler = CudaTranspiler::new_generic();
        let cuda = transpiler.transpile_generic_kernel(&func).unwrap();

        assert!(cuda.contains("extern \"C\" __global__"), "Should be global kernel: {}", cuda);
        assert!(cuda.contains("for (int i = 0; i < n; i++)"), "Should have for loop: {}", cuda);

        println!("Generated kernel with loop:\n{}", cuda);
    }

    #[test]
    fn test_persistent_kernel_pattern() {
        // Test the pattern used for ring/actor kernels
        let transpiler = CudaTranspiler::with_mode(ValidationMode::RingKernel);

        let expr: Expr = parse_quote! {
            while !should_terminate {
                if has_message {
                    process_message();
                }
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("while (!(should_terminate))"), "Should have persistent loop: {}", result);
        assert!(result.contains("if (has_message)"), "Should have message check: {}", result);

        println!("Generated persistent kernel pattern:\n{}", result);
    }

    // ==================== Shared Memory Tests ====================

    #[test]
    fn test_shared_tile_declaration() {
        use crate::shared::{SharedMemoryDecl, SharedMemoryConfig};

        let decl = SharedMemoryDecl::tile("tile", "float", 16, 16);
        assert_eq!(decl.to_cuda_decl(), "__shared__ float tile[16][16];");

        let mut config = SharedMemoryConfig::new();
        config.add_tile("tile", "float", 16, 16);
        assert_eq!(config.total_bytes(), 16 * 16 * 4); // 1024 bytes

        let decls = config.generate_declarations("    ");
        assert!(decls.contains("__shared__ float tile[16][16];"));
    }

    #[test]
    fn test_shared_array_declaration() {
        use crate::shared::{SharedMemoryDecl, SharedMemoryConfig};

        let decl = SharedMemoryDecl::array("buffer", "float", 256);
        assert_eq!(decl.to_cuda_decl(), "__shared__ float buffer[256];");

        let mut config = SharedMemoryConfig::new();
        config.add_array("buffer", "float", 256);
        assert_eq!(config.total_bytes(), 256 * 4); // 1024 bytes
    }

    #[test]
    fn test_shared_memory_access_expressions() {
        use crate::shared::SharedMemoryDecl;

        let tile = SharedMemoryDecl::tile("tile", "float", 16, 16);
        assert_eq!(
            tile.to_cuda_access(&["y".to_string(), "x".to_string()]),
            "tile[y][x]"
        );

        let arr = SharedMemoryDecl::array("buf", "int", 128);
        assert_eq!(
            arr.to_cuda_access(&["i".to_string()]),
            "buf[i]"
        );
    }

    #[test]
    fn test_parse_shared_tile_type() {
        use crate::shared::parse_shared_tile_type;

        let result = parse_shared_tile_type("SharedTile::<f32, 16, 16>");
        assert_eq!(result, Some(("f32".to_string(), 16, 16)));

        let result2 = parse_shared_tile_type("SharedTile<i32, 32, 8>");
        assert_eq!(result2, Some(("i32".to_string(), 32, 8)));

        let invalid = parse_shared_tile_type("Vec<f32>");
        assert_eq!(invalid, None);
    }

    #[test]
    fn test_parse_shared_array_type() {
        use crate::shared::parse_shared_array_type;

        let result = parse_shared_array_type("SharedArray::<f32, 256>");
        assert_eq!(result, Some(("f32".to_string(), 256)));

        let result2 = parse_shared_array_type("SharedArray<u32, 1024>");
        assert_eq!(result2, Some(("u32".to_string(), 1024)));

        let invalid = parse_shared_array_type("Vec<f32>");
        assert_eq!(invalid, None);
    }

    #[test]
    fn test_rust_to_cuda_element_types() {
        use crate::shared::rust_to_cuda_element_type;

        assert_eq!(rust_to_cuda_element_type("f32"), "float");
        assert_eq!(rust_to_cuda_element_type("f64"), "double");
        assert_eq!(rust_to_cuda_element_type("i32"), "int");
        assert_eq!(rust_to_cuda_element_type("u32"), "unsigned int");
        assert_eq!(rust_to_cuda_element_type("i64"), "long long");
        assert_eq!(rust_to_cuda_element_type("u64"), "unsigned long long");
        assert_eq!(rust_to_cuda_element_type("bool"), "int");
    }

    #[test]
    fn test_shared_memory_total_bytes() {
        use crate::shared::SharedMemoryConfig;

        let mut config = SharedMemoryConfig::new();
        config.add_tile("tile1", "float", 16, 16);  // 16*16*4 = 1024
        config.add_tile("tile2", "double", 8, 8);   // 8*8*8 = 512
        config.add_array("temp", "int", 64);        // 64*4 = 256

        assert_eq!(config.total_bytes(), 1024 + 512 + 256);
    }

    #[test]
    fn test_transpiler_shared_var_tracking() {
        let mut transpiler = CudaTranspiler::new_generic();

        // Manually register a shared variable
        transpiler.shared_vars.insert("tile".to_string(), SharedVarInfo {
            name: "tile".to_string(),
            is_tile: true,
            dimensions: vec![16, 16],
            element_type: "float".to_string(),
        });

        // Test that transpiler tracks it
        assert!(transpiler.shared_vars.contains_key("tile"));
        assert!(transpiler.shared_vars.get("tile").unwrap().is_tile);
    }

    #[test]
    fn test_shared_tile_get_transpilation() {
        let mut transpiler = CudaTranspiler::new_generic();

        // Register a shared tile
        transpiler.shared_vars.insert("tile".to_string(), SharedVarInfo {
            name: "tile".to_string(),
            is_tile: true,
            dimensions: vec![16, 16],
            element_type: "float".to_string(),
        });

        // Test method call transpilation
        let result = transpiler.try_transpile_shared_method_call(
            "tile",
            "get",
            &syn::punctuated::Punctuated::new()
        );

        // With no args, it should return None (args required)
        assert!(result.is_none() || result.unwrap().is_err());
    }

    #[test]
    fn test_shared_array_access() {
        let mut transpiler = CudaTranspiler::new_generic();

        // Register a shared array
        transpiler.shared_vars.insert("buffer".to_string(), SharedVarInfo {
            name: "buffer".to_string(),
            is_tile: false,
            dimensions: vec![256],
            element_type: "float".to_string(),
        });

        assert!(!transpiler.shared_vars.get("buffer").unwrap().is_tile);
        assert_eq!(transpiler.shared_vars.get("buffer").unwrap().dimensions, vec![256]);
    }

    #[test]
    fn test_full_kernel_with_shared_memory() {
        // Test that we can generate declarations correctly
        use crate::shared::SharedMemoryConfig;

        let mut config = SharedMemoryConfig::new();
        config.add_tile("smem", "float", 16, 16);

        let decls = config.generate_declarations("    ");
        assert!(decls.contains("__shared__ float smem[16][16];"));
        assert!(!config.is_empty());
    }

    // === Struct Literal Tests ===

    #[test]
    fn test_struct_literal_transpile() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            Point { x: 1.0, y: 2.0 }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("Point"), "Should contain struct name: {}", result);
        assert!(result.contains(".x ="), "Should have field x: {}", result);
        assert!(result.contains(".y ="), "Should have field y: {}", result);
        assert!(result.contains("1.0f"), "Should have value 1.0f: {}", result);
        assert!(result.contains("2.0f"), "Should have value 2.0f: {}", result);

        println!("Generated struct literal: {}", result);
    }

    #[test]
    fn test_struct_literal_with_expressions() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            Response { value: x * 2.0, id: idx as u64 }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("Response"), "Should contain struct name: {}", result);
        assert!(result.contains(".value = x * 2.0f"), "Should have computed value: {}", result);
        assert!(result.contains(".id ="), "Should have id field: {}", result);

        println!("Generated struct with expressions: {}", result);
    }

    #[test]
    fn test_struct_literal_in_return() {
        let mut transpiler = CudaTranspiler::new_generic();

        let stmt: Stmt = parse_quote! {
            return MyStruct { a: 1, b: 2.0 };
        };

        let result = transpiler.transpile_stmt(&stmt).unwrap();
        assert!(result.contains("return"), "Should have return: {}", result);
        assert!(result.contains("MyStruct"), "Should contain struct name: {}", result);

        println!("Generated return with struct: {}", result);
    }

    #[test]
    fn test_struct_literal_compound_literal_format() {
        let transpiler = CudaTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            Vec3 { x: a, y: b, z: c }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        // Check for C compound literal format: (Type){ .field = val, ... }
        assert!(result.starts_with("(Vec3){"), "Should use compound literal format: {}", result);
        assert!(result.ends_with("}"), "Should end with closing brace: {}", result);

        println!("Generated compound literal: {}", result);
    }
}
