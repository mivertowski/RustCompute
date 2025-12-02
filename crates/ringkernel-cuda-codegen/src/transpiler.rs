//! Core Rust-to-CUDA transpiler.
//!
//! This module handles the translation of Rust AST to CUDA C code.

use crate::intrinsics::{IntrinsicRegistry, StencilIntrinsic};
use crate::stencil::StencilConfig;
use crate::types::{is_grid_pos_type, TypeMapper};
use crate::{Result, TranspileError};
use quote::ToTokens;
use syn::{
    BinOp, Expr, ExprAssign, ExprBinary, ExprCall, ExprCast, ExprIf, ExprIndex, ExprLit,
    ExprMatch, ExprMethodCall, ExprParen, ExprPath, ExprReturn, ExprUnary, FnArg, ItemFn, Lit,
    Pat, ReturnType, Stmt, UnOp,
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
    /// Current indentation level.
    indent: usize,
}

impl CudaTranspiler {
    /// Create a new transpiler with stencil configuration.
    pub fn new(config: StencilConfig) -> Self {
        Self {
            config: Some(config),
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            indent: 1, // Start with 1 level for function body
        }
    }

    /// Create a new transpiler without stencil configuration.
    pub fn new_generic() -> Self {
        Self {
            config: None,
            type_mapper: TypeMapper::new(),
            intrinsics: IntrinsicRegistry::new(),
            grid_pos_vars: Vec::new(),
            indent: 1,
        }
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

        // Check if this is a GridPos method call
        if self.grid_pos_vars.contains(&receiver) {
            return self.transpile_stencil_intrinsic(&method_name, &method.args);
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
}
