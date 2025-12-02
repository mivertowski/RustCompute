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
    ExprMethodCall, ExprParen, ExprPath, ExprUnary, FnArg, ItemFn, Lit, Pat, ReturnType, Stmt,
    UnOp,
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
                let expr_str = self.transpile_expr(expr)?;

                if semi.is_some() {
                    Ok(format!("{indent}{expr_str};\n"))
                } else {
                    // Expression without semicolon (implicit return)
                    Ok(format!("{indent}return {expr_str};\n"))
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
            if call.args.is_empty() && cuda_name.ends_with("()") {
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

    /// Infer CUDA type from expression (simple heuristic).
    fn infer_cuda_type(&self, expr: &Expr) -> &'static str {
        match expr {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Float(_) => "float",
                Lit::Int(_) => "int",
                Lit::Bool(_) => "int",
                _ => "float",
            },
            Expr::Binary(_) | Expr::MethodCall(_) | Expr::Call(_) => "float",
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
}
