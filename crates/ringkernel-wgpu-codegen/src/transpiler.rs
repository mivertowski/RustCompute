//! Core Rust-to-WGSL transpiler.
//!
//! This module handles the translation of Rust AST to WGSL code.

use crate::bindings::{generate_bindings, AccessMode, BindingLayout};
use crate::handler::WgslContextMethod;
use crate::intrinsics::{IntrinsicRegistry, WgslIntrinsic};
use crate::loops::RangeInfo;
use crate::ring_kernel::RingKernelConfig;
use crate::shared::{SharedMemoryConfig, SharedMemoryDecl};
use crate::stencil::StencilConfig;
use crate::types::{
    is_grid_pos_type, is_mutable_reference, is_ring_context_type, TypeMapper, WgslType,
};
use crate::u64_workarounds::U64Helpers;
use crate::validation::ValidationMode;
use crate::{Result, TranspileError};
use quote::ToTokens;
use std::collections::HashMap;
use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprBreak, ExprCall, ExprCast, ExprContinue, ExprForLoop,
    ExprIf, ExprIndex, ExprLit, ExprLoop, ExprMatch, ExprMethodCall, ExprParen, ExprPath,
    ExprReturn, ExprStruct, ExprUnary, ExprWhile, FnArg, ItemFn, Lit, Pat, ReturnType, Stmt, UnOp,
};

/// WGSL code transpiler.
pub struct WgslTranspiler {
    /// Stencil configuration (if generating a stencil kernel).
    config: Option<StencilConfig>,
    /// Ring kernel configuration (if generating a ring kernel).
    #[allow(dead_code)]
    ring_config: Option<RingKernelConfig>,
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
    pub shared_vars: HashMap<String, SharedVarInfo>,
    /// Whether we're in ring kernel mode (enables context method inlining).
    ring_kernel_mode: bool,
    /// Whether to include u64 helper functions.
    needs_u64_helpers: bool,
    /// Whether subgroup operations are used (need extension).
    needs_subgroup_extension: bool,
    /// Workgroup size for generic kernels.
    workgroup_size: (u32, u32, u32),
    /// Collected buffer bindings.
    bindings: Vec<BindingLayout>,
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
    /// Element type (WGSL type string).
    pub element_type: String,
}

impl WgslTranspiler {
    /// Create a new transpiler with stencil configuration.
    pub fn new_stencil(config: StencilConfig) -> Self {
        Self {
            config: Some(config),
            ring_config: None,
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 1,
            validation_mode: ValidationMode::Stencil,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: HashMap::new(),
            ring_kernel_mode: false,
            needs_u64_helpers: false,
            needs_subgroup_extension: false,
            workgroup_size: (256, 1, 1),
            bindings: Vec::new(),
        }
    }

    /// Create a new transpiler without stencil configuration.
    pub fn new_generic() -> Self {
        Self {
            config: None,
            ring_config: None,
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 1,
            validation_mode: ValidationMode::Generic,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: HashMap::new(),
            ring_kernel_mode: false,
            needs_u64_helpers: false,
            needs_subgroup_extension: false,
            workgroup_size: (256, 1, 1),
            bindings: Vec::new(),
        }
    }

    /// Create a new transpiler for ring kernel generation.
    pub fn new_ring_kernel(config: RingKernelConfig) -> Self {
        Self {
            config: None,
            ring_config: Some(config.clone()),
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            context_vars: Vec::new(),
            indent: 2,
            validation_mode: ValidationMode::Generic,
            shared_memory: SharedMemoryConfig::new(),
            shared_vars: HashMap::new(),
            ring_kernel_mode: true,
            needs_u64_helpers: true, // Ring kernels typically need 64-bit counters
            needs_subgroup_extension: false,
            workgroup_size: (config.workgroup_size, 1, 1),
            bindings: Vec::new(),
        }
    }

    /// Set the workgroup size for generic kernels.
    pub fn with_workgroup_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = (x, y, z);
        self
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

        // Generate bindings from parameters
        self.collect_bindings(func)?;

        let mut output = String::new();

        // Generate buffer bindings
        output.push_str(&generate_bindings(&self.bindings));
        output.push_str("\n\n");

        // Generate shared memory declarations
        if config.use_shared_memory {
            let buffer_width = config.buffer_width();
            let buffer_height = config.buffer_height();
            output.push_str(&format!(
                "var<workgroup> tile: array<array<f32, {}>, {}>;\n\n",
                buffer_width, buffer_height
            ));
        }

        // Generate kernel signature with builtins
        output.push_str("@compute ");
        output.push_str(&config.workgroup_size_annotation());
        output.push('\n');
        output.push_str(&format!("fn {}(\n", func.sig.ident));
        output.push_str("    @builtin(local_invocation_id) local_id: vec3<u32>,\n");
        output.push_str("    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n");
        output.push_str("    @builtin(global_invocation_id) global_id: vec3<u32>\n");
        output.push_str(") {\n");

        // Generate preamble
        output.push_str(&self.generate_stencil_preamble(&config));

        // Generate function body
        let body = self.transpile_block(&func.block)?;
        output.push_str(&body);

        output.push_str("}\n");

        Ok(output)
    }

    /// Generate stencil kernel preamble.
    fn generate_stencil_preamble(&self, config: &StencilConfig) -> String {
        let buffer_width = config.buffer_width();
        let mut preamble = String::new();

        preamble.push_str("    // Thread indices\n");
        preamble.push_str("    let lx = local_id.x;\n");
        preamble.push_str("    let ly = local_id.y;\n");
        preamble.push_str(&format!("    let buffer_width = {}u;\n", buffer_width));
        preamble.push('\n');

        // Bounds check
        preamble.push_str(&format!(
            "    if (lx >= {}u || ly >= {}u) {{ return; }}\n\n",
            config.tile_width, config.tile_height
        ));

        // Calculate buffer index with halo offset
        preamble.push_str(&format!(
            "    let idx = (ly + {}u) * buffer_width + (lx + {}u);\n\n",
            config.halo, config.halo
        ));

        preamble
    }

    /// Transpile a generic (non-stencil) kernel function.
    pub fn transpile_global_kernel(&mut self, func: &ItemFn) -> Result<String> {
        // Collect bindings from parameters
        self.collect_bindings(func)?;

        let mut output = String::new();

        // Add extensions if needed
        if self.needs_subgroup_extension {
            output.push_str("enable chromium_experimental_subgroups;\n\n");
        }

        // Generate buffer bindings
        output.push_str(&generate_bindings(&self.bindings));
        output.push_str("\n\n");

        // Generate shared memory declarations
        if !self.shared_memory.declarations.is_empty() {
            output.push_str(&self.shared_memory.to_wgsl());
            output.push_str("\n\n");
        }

        // Generate kernel signature
        output.push_str("@compute ");
        output.push_str(&format!(
            "@workgroup_size({}, {}, {})\n",
            self.workgroup_size.0, self.workgroup_size.1, self.workgroup_size.2
        ));
        output.push_str(&format!("fn {}(\n", func.sig.ident));
        output.push_str("    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,\n");
        output.push_str("    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n");
        output.push_str("    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,\n");
        output.push_str("    @builtin(num_workgroups) num_workgroups: vec3<u32>\n");
        output.push_str(") {\n");

        // Generate function body
        let body = self.transpile_block(&func.block)?;
        output.push_str(&body);

        output.push_str("}\n");

        Ok(output)
    }

    /// Transpile a handler function into a ring kernel.
    pub fn transpile_ring_kernel(
        &mut self,
        handler: &ItemFn,
        config: &RingKernelConfig,
    ) -> Result<String> {
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

        self.ring_kernel_mode = true;
        self.needs_u64_helpers = true;

        let mut output = String::new();

        // Generate 64-bit helper functions
        output.push_str(&U64Helpers::generate_all());
        output.push_str("\n\n");

        // Generate ControlBlock struct
        output.push_str(&crate::ring_kernel::generate_control_block_struct(config));
        output.push_str("\n\n");

        // Generate bindings
        output.push_str(crate::ring_kernel::generate_ring_kernel_bindings());
        output.push_str("\n\n");

        // Generate kernel
        output.push_str("@compute ");
        output.push_str(&config.workgroup_size_annotation());
        output.push('\n');
        output.push_str(&format!("fn ring_kernel_{}(\n", config.name));
        output.push_str("    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,\n");
        output.push_str("    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n");
        output.push_str("    @builtin(global_invocation_id) global_invocation_id: vec3<u32>\n");
        output.push_str(") {\n");

        // Generate preamble (activation/termination checks)
        output.push_str(crate::ring_kernel::generate_ring_kernel_preamble());
        output.push_str("\n\n");

        // Generate handler body
        output.push_str("    // === USER HANDLER CODE ===\n");
        self.indent = 1;
        let handler_body = self.transpile_block(&handler.block)?;
        output.push_str(&handler_body);
        output.push_str("    // === END HANDLER CODE ===\n\n");

        // Update message counter
        output.push_str("    // Update message counter\n");
        output.push_str(
            "    atomic_inc_u64(&control.messages_processed_lo, &control.messages_processed_hi);\n",
        );

        output.push_str("}\n");

        Ok(output)
    }

    /// Collect buffer bindings from function parameters.
    fn collect_bindings(&mut self, func: &ItemFn) -> Result<()> {
        let mut binding_idx = 0u32;

        for param in &func.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                // Skip GridPos and RingContext parameters
                if is_grid_pos_type(&pat_type.ty) || is_ring_context_type(&pat_type.ty) {
                    continue;
                }

                let param_name = match pat_type.pat.as_ref() {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    _ => continue,
                };

                let wgsl_type = self
                    .type_mapper
                    .map_type(&pat_type.ty)
                    .map_err(TranspileError::Type)?;
                let is_mutable = is_mutable_reference(&pat_type.ty);

                // Determine if this is a buffer or scalar
                match &wgsl_type {
                    WgslType::Ptr { inner, .. } => {
                        let access = if is_mutable {
                            AccessMode::ReadWrite
                        } else {
                            AccessMode::Read
                        };
                        self.bindings.push(BindingLayout::new(
                            0,
                            binding_idx,
                            &param_name,
                            WgslType::Array {
                                element: inner.clone(),
                                size: None,
                            },
                            access,
                        ));
                        binding_idx += 1;
                    }
                    WgslType::Array { .. } => {
                        let access = if is_mutable {
                            AccessMode::ReadWrite
                        } else {
                            AccessMode::Read
                        };
                        self.bindings.push(BindingLayout::new(
                            0,
                            binding_idx,
                            &param_name,
                            wgsl_type.clone(),
                            access,
                        ));
                        binding_idx += 1;
                    }
                    // Scalars are typically passed via uniforms
                    _ => {
                        self.bindings.push(BindingLayout::uniform(
                            binding_idx,
                            &param_name,
                            wgsl_type.clone(),
                        ));
                        binding_idx += 1;
                    }
                }
            }
        }

        Ok(())
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
                    self.shared_vars.insert(
                        var_name.clone(),
                        SharedVarInfo {
                            name: var_name.clone(),
                            is_tile: shared_decl.dimensions.len() == 2,
                            dimensions: shared_decl
                                .dimensions
                                .iter()
                                .map(|&d| d as usize)
                                .collect(),
                            element_type: shared_decl.element_type.to_wgsl(),
                        },
                    );
                    self.shared_memory.add(shared_decl.clone());
                    return Ok(format!(
                        "{indent}// shared memory: {} declared at module scope\n",
                        var_name
                    ));
                }

                // Get initializer
                if let Some(init) = &local.init {
                    let expr_str = self.transpile_expr(&init.expr)?;
                    let type_str = self.infer_wgsl_type(&init.expr);

                    Ok(format!(
                        "{indent}var {var_name}: {type_str} = {expr_str};\n"
                    ))
                } else {
                    Ok(format!("{indent}var {var_name}: f32;\n"))
                }
            }
            Stmt::Expr(expr, semi) => {
                let indent = self.indent_str();

                // Handle early return pattern in if statements
                if let Expr::If(if_expr) = expr {
                    if let Some(Stmt::Expr(Expr::Return(_), _)) = if_expr.then_branch.stmts.first()
                    {
                        if if_expr.then_branch.stmts.len() == 1 && if_expr.else_branch.is_none() {
                            let expr_str = self.transpile_expr(expr)?;
                            return Ok(format!("{indent}{expr_str}\n"));
                        }
                    }
                }

                let expr_str = self.transpile_expr(expr)?;

                if semi.is_some()
                    || matches!(expr, Expr::Return(_))
                    || expr_str.starts_with("return")
                    || expr_str.starts_with("if (")
                {
                    Ok(format!("{indent}{expr_str};\n"))
                } else {
                    Ok(format!("{indent}return {expr_str};\n"))
                }
            }
            Stmt::Item(_) => Err(TranspileError::Unsupported("Item in function body".into())),
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
                if let Some(Stmt::Expr(expr, None)) = block.block.stmts.last() {
                    self.transpile_expr(expr)
                } else {
                    Err(TranspileError::Unsupported(
                        "Complex block expression".into(),
                    ))
                }
            }
            Expr::Field(field) => {
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
            Expr::Reference(ref_expr) => {
                // In WGSL, we often need &var for atomics
                let inner = self.transpile_expr(&ref_expr.expr)?;
                Ok(format!("&{inner}"))
            }
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
                // WGSL float literals don't need suffix
                let num = s.trim_end_matches("f32").trim_end_matches("f64");
                // Ensure there's a decimal point
                if num.contains('.') {
                    Ok(num.to_string())
                } else {
                    Ok(format!("{}.0", num))
                }
            }
            Lit::Int(i) => {
                let s = i.to_string();
                // Check for unsigned suffix
                if s.ends_with("u32") || s.ends_with("usize") {
                    Ok(format!(
                        "{}u",
                        s.trim_end_matches("u32").trim_end_matches("usize")
                    ))
                } else if s.ends_with("i32") || s.ends_with("isize") {
                    Ok(format!(
                        "{}i",
                        s.trim_end_matches("i32").trim_end_matches("isize")
                    ))
                } else {
                    // Default to signed int
                    Ok(s)
                }
            }
            Lit::Bool(b) => Ok(if b.value { "true" } else { "false" }.to_string()),
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
            Ok(segments.join("_"))
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
            return self.transpile_intrinsic_call(intrinsic, &call.args);
        }

        // Regular function call
        let args: Vec<String> = call
            .args
            .iter()
            .map(|a| self.transpile_expr(a))
            .collect::<Result<_>>()?;

        Ok(format!("{}({})", func, args.join(", ")))
    }

    /// Transpile an intrinsic call.
    fn transpile_intrinsic_call(
        &self,
        intrinsic: WgslIntrinsic,
        args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    ) -> Result<String> {
        let wgsl_name = intrinsic.to_wgsl();

        // Check for value intrinsics (builtins accessed as variables)
        match intrinsic {
            WgslIntrinsic::LocalInvocationIdX
            | WgslIntrinsic::LocalInvocationIdY
            | WgslIntrinsic::LocalInvocationIdZ
            | WgslIntrinsic::WorkgroupIdX
            | WgslIntrinsic::WorkgroupIdY
            | WgslIntrinsic::WorkgroupIdZ
            | WgslIntrinsic::GlobalInvocationIdX
            | WgslIntrinsic::GlobalInvocationIdY
            | WgslIntrinsic::GlobalInvocationIdZ
            | WgslIntrinsic::NumWorkgroupsX
            | WgslIntrinsic::NumWorkgroupsY
            | WgslIntrinsic::NumWorkgroupsZ => {
                // These are variable accesses, not function calls
                return Ok(format!("i32({})", wgsl_name));
            }
            WgslIntrinsic::WorkgroupSizeX => {
                return Ok(format!("i32({}u)", self.workgroup_size.0));
            }
            WgslIntrinsic::WorkgroupSizeY => {
                return Ok(format!("i32({}u)", self.workgroup_size.1));
            }
            WgslIntrinsic::WorkgroupSizeZ => {
                return Ok(format!("i32({}u)", self.workgroup_size.2));
            }
            WgslIntrinsic::WorkgroupBarrier | WgslIntrinsic::StorageBarrier => {
                // Zero-arg function intrinsics
                return Ok(wgsl_name.to_string());
            }
            WgslIntrinsic::SubgroupInvocationId | WgslIntrinsic::SubgroupSize => {
                // Subgroup builtins
                return Ok(wgsl_name.to_string());
            }
            _ => {}
        }

        // Function intrinsics with arguments
        let transpiled_args: Vec<String> = args
            .iter()
            .map(|a| self.transpile_expr(a))
            .collect::<Result<_>>()?;

        Ok(format!("{}({})", wgsl_name, transpiled_args.join(", ")))
    }

    /// Transpile a method call.
    fn transpile_method_call(&self, method: &ExprMethodCall) -> Result<String> {
        let receiver = self.transpile_expr(&method.receiver)?;
        let method_name = method.method.to_string();

        // Check if this is a SharedTile/SharedArray method call
        if let Some(result) =
            self.try_transpile_shared_method_call(&receiver, &method_name, &method.args)
        {
            return result;
        }

        // Check if this is a RingContext method call
        if self.ring_kernel_mode && self.context_vars.contains(&receiver) {
            return self.transpile_context_method(&method_name, &method.args);
        }

        // Check if this is a GridPos method call
        if self.grid_pos_vars.contains(&receiver) {
            return self.transpile_stencil_intrinsic(&method_name, &method.args);
        }

        // Check for f32 math methods
        if let Some(intrinsic) = self.intrinsics.lookup(&method_name) {
            let wgsl_name = intrinsic.to_wgsl();
            let args: Vec<String> = std::iter::once(receiver)
                .chain(
                    method
                        .args
                        .iter()
                        .map(|a| self.transpile_expr(a).unwrap_or_default()),
                )
                .collect();

            return Ok(format!("{}({})", wgsl_name, args.join(", ")));
        }

        // Regular method call
        let args: Vec<String> = method
            .args
            .iter()
            .map(|a| self.transpile_expr(a))
            .collect::<Result<_>>()?;

        Ok(format!("{}.{}({})", receiver, method_name, args.join(", ")))
    }

    /// Transpile a RingContext method call.
    fn transpile_context_method(
        &self,
        method: &str,
        args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    ) -> Result<String> {
        let ctx_method = WgslContextMethod::from_name(method).ok_or_else(|| {
            TranspileError::Unsupported(format!("Unknown context method: {}", method))
        })?;

        let wgsl_args: Vec<String> = args
            .iter()
            .map(|a| self.transpile_expr(a).unwrap_or_default())
            .collect();

        match ctx_method {
            WgslContextMethod::AtomicAdd
            | WgslContextMethod::AtomicLoad
            | WgslContextMethod::AtomicStore => Ok(format!(
                "{}({})",
                ctx_method.to_wgsl(),
                wgsl_args.join(", ")
            )),
            _ => Ok(ctx_method.to_wgsl().to_string()),
        }
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

        let buffer_width = config.buffer_width();

        match method {
            "idx" => Ok("idx".to_string()),
            "x" => Ok("i32(lx)".to_string()),
            "y" => Ok("i32(ly)".to_string()),
            "north" => {
                if args.is_empty() {
                    return Err(TranspileError::Unsupported(
                        "north() requires buffer argument".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                Ok(format!("{buffer}[idx - {buffer_width}u]"))
            }
            "south" => {
                if args.is_empty() {
                    return Err(TranspileError::Unsupported(
                        "south() requires buffer argument".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                Ok(format!("{buffer}[idx + {buffer_width}u]"))
            }
            "east" => {
                if args.is_empty() {
                    return Err(TranspileError::Unsupported(
                        "east() requires buffer argument".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                Ok(format!("{buffer}[idx + 1u]"))
            }
            "west" => {
                if args.is_empty() {
                    return Err(TranspileError::Unsupported(
                        "west() requires buffer argument".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                Ok(format!("{buffer}[idx - 1u]"))
            }
            "at" => {
                if args.len() < 3 {
                    return Err(TranspileError::Unsupported(
                        "at() requires buffer, dx, dy arguments".into(),
                    ));
                }
                let buffer = self.transpile_expr(&args[0])?;
                let dx = self.transpile_expr(&args[1])?;
                let dy = self.transpile_expr(&args[2])?;
                Ok(format!(
                    "{buffer}[idx + u32(({dy}) * i32({buffer_width}u) + ({dx}))]"
                ))
            }
            _ => Err(TranspileError::Unsupported(format!(
                "Unknown stencil intrinsic: {}",
                method
            ))),
        }
    }

    /// Transpile an if expression.
    fn transpile_if(&self, if_expr: &ExprIf) -> Result<String> {
        let cond = self.transpile_expr(&if_expr.cond)?;

        // Check for early return pattern
        if let Some(Stmt::Expr(Expr::Return(ret), _)) = if_expr.then_branch.stmts.first() {
            if if_expr.then_branch.stmts.len() == 1 && if_expr.else_branch.is_none() {
                if ret.expr.is_none() {
                    return Ok(format!("if ({cond}) {{ return; }}"));
                }
                let ret_val = self.transpile_expr(ret.expr.as_ref().unwrap())?;
                return Ok(format!("if ({cond}) {{ return {ret_val}; }}"));
            }
        }

        // Handle if-else as select when possible
        if let Some((_, else_branch)) = &if_expr.else_branch {
            if let (Some(Stmt::Expr(then_expr, None)), Expr::Block(else_block)) =
                (if_expr.then_branch.stmts.last(), else_branch.as_ref())
            {
                if let Some(Stmt::Expr(else_expr, None)) = else_block.block.stmts.last() {
                    let then_str = self.transpile_expr(then_expr)?;
                    let else_str = self.transpile_expr(else_expr)?;
                    return Ok(format!("select({else_str}, {then_str}, {cond})"));
                }
            }

            // else if chain
            if let Expr::If(else_if) = else_branch.as_ref() {
                let then_body = self.transpile_if_body(&if_expr.then_branch)?;
                let else_part = self.transpile_if(else_if)?;
                return Ok(format!("if ({cond}) {{{then_body}}} else {else_part}"));
            } else if let Expr::Block(else_block) = else_branch.as_ref() {
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
        let wgsl_type = self
            .type_mapper
            .map_type(&cast.ty)
            .map_err(TranspileError::Type)?;
        Ok(format!("{}({})", wgsl_type.to_wgsl(), expr))
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

    /// Transpile a struct literal.
    fn transpile_struct_literal(&self, struct_expr: &ExprStruct) -> Result<String> {
        let type_name = struct_expr
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("_");

        let mut fields = Vec::new();
        for field in &struct_expr.fields {
            let field_name = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(idx) => idx.index.to_string(),
            };
            let value = self.transpile_expr(&field.expr)?;
            fields.push(format!("{}: {}", field_name, value));
        }

        if struct_expr.rest.is_some() {
            return Err(TranspileError::Unsupported(
                "Struct update syntax (..base) is not supported in WGSL".into(),
            ));
        }

        Ok(format!("{}({})", type_name, fields.join(", ")))
    }

    // === Loop Transpilation ===

    /// Transpile a for loop.
    fn transpile_for_loop(&self, for_loop: &ExprForLoop) -> Result<String> {
        if !self.validation_mode.allows_loops() {
            return Err(TranspileError::Unsupported(
                "Loops are not allowed in stencil kernels".into(),
            ));
        }

        let var_name = extract_loop_var(&for_loop.pat)
            .ok_or_else(|| TranspileError::Unsupported("Complex pattern in for loop".into()))?;

        let header = match for_loop.expr.as_ref() {
            Expr::Range(range) => {
                let range_info = RangeInfo::from_range(range, |e| self.transpile_expr(e));
                range_info.to_wgsl_for_header(&var_name)
            }
            _ => {
                return Err(TranspileError::Unsupported(
                    "Only range expressions are supported in for loops".into(),
                ));
            }
        };

        let body = self.transpile_loop_body(&for_loop.body)?;
        Ok(format!("{header} {{\n{body}}}"))
    }

    /// Transpile a while loop.
    fn transpile_while_loop(&self, while_loop: &ExprWhile) -> Result<String> {
        if !self.validation_mode.allows_loops() {
            return Err(TranspileError::Unsupported(
                "Loops are not allowed in stencil kernels".into(),
            ));
        }

        let condition = self.transpile_expr(&while_loop.cond)?;
        let body = self.transpile_loop_body(&while_loop.body)?;
        Ok(format!("while ({condition}) {{\n{body}}}"))
    }

    /// Transpile an infinite loop.
    fn transpile_infinite_loop(&self, loop_expr: &ExprLoop) -> Result<String> {
        if !self.validation_mode.allows_loops() {
            return Err(TranspileError::Unsupported(
                "Loops are not allowed in stencil kernels".into(),
            ));
        }

        let body = self.transpile_loop_body(&loop_expr.body)?;
        Ok(format!("loop {{\n{body}}}"))
    }

    /// Transpile a break expression.
    fn transpile_break(&self, break_expr: &ExprBreak) -> Result<String> {
        if break_expr.label.is_some() {
            return Err(TranspileError::Unsupported(
                "Labeled break is not supported in WGSL".into(),
            ));
        }
        if break_expr.expr.is_some() {
            return Err(TranspileError::Unsupported(
                "Break with value is not supported in WGSL".into(),
            ));
        }
        Ok("break".to_string())
    }

    /// Transpile a continue expression.
    fn transpile_continue(&self, cont_expr: &ExprContinue) -> Result<String> {
        if cont_expr.label.is_some() {
            return Err(TranspileError::Unsupported(
                "Labeled continue is not supported in WGSL".into(),
            ));
        }
        Ok("continue".to_string())
    }

    /// Transpile a loop body.
    fn transpile_loop_body(&self, block: &syn::Block) -> Result<String> {
        let mut output = String::new();
        let inner_indent = "    ".repeat(self.indent + 1);

        for stmt in &block.stmts {
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
                        let type_str = self.infer_wgsl_type(&init.expr);
                        output.push_str(&format!(
                            "{inner_indent}var {var_name}: {type_str} = {expr_str};\n"
                        ));
                    } else {
                        output.push_str(&format!("{inner_indent}var {var_name}: f32;\n"));
                    }
                }
                Stmt::Expr(expr, _semi) => {
                    let expr_str = self.transpile_expr(expr)?;
                    output.push_str(&format!("{inner_indent}{expr_str};\n"));
                }
                _ => {
                    return Err(TranspileError::Unsupported(
                        "Unsupported statement in loop body".into(),
                    ));
                }
            }
        }

        let closing_indent = "    ".repeat(self.indent);
        output.push_str(&closing_indent);

        Ok(output)
    }

    /// Transpile a match expression to switch.
    fn transpile_match(&self, match_expr: &ExprMatch) -> Result<String> {
        let scrutinee = self.transpile_expr(&match_expr.expr)?;
        let mut output = format!("switch ({scrutinee}) {{\n");

        for arm in &match_expr.arms {
            let case_label = self.transpile_match_pattern(&arm.pat)?;

            if case_label == "default" || case_label.starts_with("/*") {
                output.push_str("    default: {\n");
            } else {
                output.push_str(&format!("    case {case_label}: {{\n"));
            }

            match arm.body.as_ref() {
                Expr::Block(block) => {
                    for stmt in &block.block.stmts {
                        let stmt_str = self.transpile_stmt_inline(stmt)?;
                        output.push_str(&format!("        {stmt_str}\n"));
                    }
                }
                _ => {
                    let body = self.transpile_expr(&arm.body)?;
                    output.push_str(&format!("        {body};\n"));
                }
            }

            output.push_str("    }\n");
        }

        output.push('}');
        Ok(output)
    }

    /// Transpile a match pattern.
    fn transpile_match_pattern(&self, pat: &Pat) -> Result<String> {
        match pat {
            Pat::Lit(pat_lit) => match &pat_lit.lit {
                Lit::Int(i) => Ok(i.to_string()),
                Lit::Bool(b) => Ok(if b.value { "true" } else { "false" }.to_string()),
                _ => Err(TranspileError::Unsupported(
                    "Non-integer literal in match pattern".into(),
                )),
            },
            Pat::Wild(_) => Ok("default".to_string()),
            Pat::Ident(ident) => Ok(format!("/* {} */ default", ident.ident)),
            Pat::Or(pat_or) => {
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

    /// Transpile a statement inline.
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
                    let type_str = self.infer_wgsl_type(&init.expr);
                    Ok(format!("var {var_name}: {type_str} = {expr_str};"))
                } else {
                    Ok(format!("var {var_name}: f32;"))
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
            _ => Err(TranspileError::Unsupported(
                "Unsupported statement in match arm".into(),
            )),
        }
    }

    // === Shared Memory Support ===

    /// Try to parse a shared memory declaration.
    fn try_parse_shared_declaration(
        &self,
        local: &syn::Local,
        var_name: &str,
    ) -> Result<Option<SharedMemoryDecl>> {
        if let Pat::Type(pat_type) = &local.pat {
            let type_str = pat_type.ty.to_token_stream().to_string();
            return self.parse_shared_type(&type_str, var_name);
        }

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

    /// Parse a type string for shared memory info.
    fn parse_shared_type(
        &self,
        type_str: &str,
        var_name: &str,
    ) -> Result<Option<SharedMemoryDecl>> {
        let type_str = type_str
            .replace(" :: ", "::")
            .replace(" ::", "::")
            .replace(":: ", "::");

        if type_str.contains("SharedTile") {
            if let Some(start) = type_str.find('<') {
                if let Some(end) = type_str.rfind('>') {
                    let params = &type_str[start + 1..end];
                    let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

                    if parts.len() >= 3 {
                        let rust_type = parts[0];
                        let width: u32 = parts[1].parse().map_err(|_| {
                            TranspileError::Unsupported("Invalid SharedTile width".into())
                        })?;
                        let height: u32 = parts[2].parse().map_err(|_| {
                            TranspileError::Unsupported("Invalid SharedTile height".into())
                        })?;

                        let wgsl_type = rust_type_to_wgsl(rust_type);
                        return Ok(Some(SharedMemoryDecl::new_2d(
                            var_name, wgsl_type, width, height,
                        )));
                    }
                }
            }
        }

        if type_str.contains("SharedArray") {
            if let Some(start) = type_str.find('<') {
                if let Some(end) = type_str.rfind('>') {
                    let params = &type_str[start + 1..end];
                    let parts: Vec<&str> = params.split(',').map(|s| s.trim()).collect();

                    if parts.len() >= 2 {
                        let rust_type = parts[0];
                        let size: u32 = parts[1].parse().map_err(|_| {
                            TranspileError::Unsupported("Invalid SharedArray size".into())
                        })?;

                        let wgsl_type = rust_type_to_wgsl(rust_type);
                        return Ok(Some(SharedMemoryDecl::new_1d(var_name, wgsl_type, size)));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Try to transpile a shared memory method call.
    fn try_transpile_shared_method_call(
        &self,
        receiver: &str,
        method_name: &str,
        args: &syn::punctuated::Punctuated<Expr, syn::token::Comma>,
    ) -> Option<Result<String>> {
        let shared_info = self.shared_vars.get(receiver)?;

        match method_name {
            "get" => {
                if shared_info.is_tile {
                    if args.len() >= 2 {
                        let x = self.transpile_expr(&args[0]).ok()?;
                        let y = self.transpile_expr(&args[1]).ok()?;
                        Some(Ok(format!("{}[{}][{}]", receiver, y, x)))
                    } else {
                        Some(Err(TranspileError::Unsupported(
                            "SharedTile.get requires x and y arguments".into(),
                        )))
                    }
                } else if !args.is_empty() {
                    let idx = self.transpile_expr(&args[0]).ok()?;
                    Some(Ok(format!("{}[{}]", receiver, idx)))
                } else {
                    Some(Err(TranspileError::Unsupported(
                        "SharedArray.get requires index argument".into(),
                    )))
                }
            }
            "set" => {
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
                } else if args.len() >= 2 {
                    let idx = self.transpile_expr(&args[0]).ok()?;
                    let val = self.transpile_expr(&args[1]).ok()?;
                    Some(Ok(format!("{}[{}] = {}", receiver, idx, val)))
                } else {
                    Some(Err(TranspileError::Unsupported(
                        "SharedArray.set requires index and value arguments".into(),
                    )))
                }
            }
            "width" if shared_info.is_tile => Some(Ok(shared_info.dimensions[1].to_string())),
            "height" if shared_info.is_tile => Some(Ok(shared_info.dimensions[0].to_string())),
            "size" => {
                let total: usize = shared_info.dimensions.iter().product();
                Some(Ok(total.to_string()))
            }
            _ => None,
        }
    }

    /// Infer WGSL type from expression.
    fn infer_wgsl_type(&self, expr: &Expr) -> &'static str {
        match expr {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Float(_) => "f32",
                Lit::Int(i) => {
                    let s = i.to_string();
                    if s.ends_with("u32") || s.ends_with("usize") {
                        "u32"
                    } else {
                        "i32"
                    }
                }
                Lit::Bool(_) => "bool",
                _ => "f32",
            },
            Expr::Binary(bin) => {
                let left_type = self.infer_wgsl_type(&bin.left);
                let right_type = self.infer_wgsl_type(&bin.right);
                if left_type == "i32" && right_type == "i32" {
                    "i32"
                } else if left_type == "u32" && right_type == "u32" {
                    "u32"
                } else {
                    "f32"
                }
            }
            Expr::Call(call) => {
                if let Ok(func) = self.transpile_expr(&call.func) {
                    if let Some(
                        WgslIntrinsic::LocalInvocationIdX
                        | WgslIntrinsic::WorkgroupIdX
                        | WgslIntrinsic::GlobalInvocationIdX,
                    ) = self.intrinsics.lookup(&func)
                    {
                        return "i32";
                    }
                }
                "f32"
            }
            Expr::Cast(cast) => {
                if let Ok(wgsl_type) = self.type_mapper.map_type(&cast.ty) {
                    match wgsl_type {
                        WgslType::I32 => return "i32",
                        WgslType::U32 => return "u32",
                        WgslType::F32 => return "f32",
                        WgslType::Bool => return "bool",
                        _ => {}
                    }
                }
                "f32"
            }
            _ => "f32",
        }
    }
}

/// Extract loop variable name from pattern.
fn extract_loop_var(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Ident(ident) => Some(ident.ident.to_string()),
        _ => None,
    }
}

/// Convert Rust primitive type to WGSL type.
fn rust_type_to_wgsl(rust_type: &str) -> WgslType {
    match rust_type {
        "f32" => WgslType::F32,
        "f64" => WgslType::F32, // WGSL doesn't have f64
        "i32" => WgslType::I32,
        "u32" => WgslType::U32,
        "bool" => WgslType::Bool,
        _ => WgslType::F32,
    }
}

/// Transpile a function to WGSL without kernel attributes.
pub fn transpile_function(func: &ItemFn) -> Result<String> {
    let mut transpiler = WgslTranspiler::new_generic();

    let name = func.sig.ident.to_string();

    let mut params = Vec::new();
    for param in &func.sig.inputs {
        if let FnArg::Typed(pat_type) = param {
            let param_name = match pat_type.pat.as_ref() {
                Pat::Ident(ident) => ident.ident.to_string(),
                _ => continue,
            };

            let wgsl_type = transpiler
                .type_mapper
                .map_type(&pat_type.ty)
                .map_err(TranspileError::Type)?;
            params.push(format!("{}: {}", param_name, wgsl_type.to_wgsl()));
        }
    }

    let return_type = match &func.sig.output {
        ReturnType::Default => "".to_string(),
        ReturnType::Type(_, ty) => {
            let wgsl_type = transpiler
                .type_mapper
                .map_type(ty)
                .map_err(TranspileError::Type)?;
            format!(" -> {}", wgsl_type.to_wgsl())
        }
    };

    let body = transpiler.transpile_block(&func.block)?;

    Ok(format!(
        "fn {name}({}){return_type} {{\n{body}}}\n",
        params.join(", ")
    ))
}

// === Range Info Extension for WGSL ===

impl RangeInfo {
    /// Create from a syn::ExprRange.
    pub fn from_range<F>(range: &syn::ExprRange, transpile: F) -> Self
    where
        F: Fn(&Expr) -> Result<String>,
    {
        let start = range.start.as_ref().and_then(|e| transpile(e).ok());
        let end = range.end.as_ref().and_then(|e| transpile(e).ok());
        let inclusive = matches!(range.limits, syn::RangeLimits::Closed(_));

        Self::new(start, end, inclusive)
    }

    /// Generate WGSL for loop header.
    pub fn to_wgsl_for_header(&self, var_name: &str) -> String {
        let start = self.start_or_default();
        let end = self.end.as_deref().unwrap_or("/* unbounded */");
        let op = if self.inclusive { "<=" } else { "<" };

        format!(
            "for (var {var_name}: i32 = {start}; {var_name} {op} {end}; {var_name} = {var_name} + 1)"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_simple_arithmetic() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote!(a + b * 2.0);
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "a + b * 2.0");
    }

    #[test]
    fn test_let_binding() {
        let mut transpiler = WgslTranspiler::new_generic();

        let stmt: Stmt = parse_quote!(let x = a + b;);
        let result = transpiler.transpile_stmt(&stmt).unwrap();
        assert!(result.contains("var x: f32 = a + b;"));
    }

    #[test]
    fn test_array_index() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote!(data[idx]);
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "data[idx]");
    }

    #[test]
    fn test_intrinsic_thread_idx() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote!(thread_idx_x());
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("local_invocation_id.x"));
    }

    #[test]
    fn test_for_loop() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            for i in 0..n {
                data[i] = 0.0;
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("for (var i: i32 = 0; i < n; i = i + 1)"));
    }

    #[test]
    fn test_while_loop() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            while x < 10 {
                x += 1;
            }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("while (x < 10)"));
    }

    #[test]
    fn test_early_return() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote! {
            if idx >= n { return; }
        };

        let result = transpiler.transpile_expr(&expr).unwrap();
        assert!(result.contains("if (idx >= n)"));
        assert!(result.contains("return;"));
    }

    #[test]
    fn test_sync_threads() {
        let transpiler = WgslTranspiler::new_generic();

        let expr: Expr = parse_quote!(sync_threads());
        let result = transpiler.transpile_expr(&expr).unwrap();
        assert_eq!(result, "workgroupBarrier()");
    }
}
