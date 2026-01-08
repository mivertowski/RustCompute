//! Fuzz target for CUDA code generation.
//!
//! Tests the CUDA transpiler with random Rust-like input to find crashes
//! or panics in the code generation logic.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

/// Represents a fuzzable kernel function structure.
#[derive(Debug, Arbitrary)]
struct FuzzKernel {
    name: FuzzIdent,
    params: Vec<FuzzParam>,
    body: Vec<FuzzStmt>,
}

#[derive(Debug, Arbitrary)]
struct FuzzIdent {
    len: u8,
}

impl FuzzIdent {
    fn to_string(&self) -> String {
        let len = (self.len % 16) as usize + 1;
        format!("ident_{}", len)
    }
}

#[derive(Debug, Arbitrary)]
struct FuzzParam {
    name: FuzzIdent,
    ty: FuzzType,
    is_ref: bool,
    is_mut: bool,
}

#[derive(Debug, Arbitrary)]
enum FuzzType {
    F32,
    F64,
    I32,
    U32,
    Bool,
    SliceF32,
    SliceI32,
}

impl FuzzType {
    fn to_rust_type(&self) -> &'static str {
        match self {
            FuzzType::F32 => "f32",
            FuzzType::F64 => "f64",
            FuzzType::I32 => "i32",
            FuzzType::U32 => "u32",
            FuzzType::Bool => "bool",
            FuzzType::SliceF32 => "[f32]",
            FuzzType::SliceI32 => "[i32]",
        }
    }
}

#[derive(Debug, Arbitrary)]
enum FuzzStmt {
    Let {
        name: FuzzIdent,
        value: FuzzExpr,
    },
    Assign {
        target: FuzzIdent,
        value: FuzzExpr,
    },
    If {
        cond: FuzzExpr,
        then_body: Vec<FuzzStmt>,
    },
    IfElse {
        cond: FuzzExpr,
        then_body: Vec<FuzzStmt>,
        else_body: Vec<FuzzStmt>,
    },
    For {
        var: FuzzIdent,
        start: i32,
        end: i32,
        body: Vec<FuzzStmt>,
    },
    While {
        cond: FuzzExpr,
        body: Vec<FuzzStmt>,
    },
    Return(Option<FuzzExpr>),
    Expr(FuzzExpr),
}

#[derive(Debug, Arbitrary)]
enum FuzzExpr {
    Literal(FuzzLiteral),
    Ident(FuzzIdent),
    Binary {
        op: FuzzBinOp,
        left: Box<FuzzExpr>,
        right: Box<FuzzExpr>,
    },
    Unary {
        op: FuzzUnaryOp,
        operand: Box<FuzzExpr>,
    },
    Call {
        func: FuzzIntrinsic,
        args: Vec<FuzzExpr>,
    },
    Index {
        array: FuzzIdent,
        index: Box<FuzzExpr>,
    },
    Cast {
        expr: Box<FuzzExpr>,
        ty: FuzzType,
    },
}

#[derive(Debug, Arbitrary)]
enum FuzzLiteral {
    Int(i32),
    Float(f32),
    Bool(bool),
}

#[derive(Debug, Arbitrary)]
enum FuzzBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    Shl,
    Shr,
}

impl FuzzBinOp {
    fn to_str(&self) -> &'static str {
        match self {
            FuzzBinOp::Add => "+",
            FuzzBinOp::Sub => "-",
            FuzzBinOp::Mul => "*",
            FuzzBinOp::Div => "/",
            FuzzBinOp::Rem => "%",
            FuzzBinOp::And => "&",
            FuzzBinOp::Or => "|",
            FuzzBinOp::Xor => "^",
            FuzzBinOp::Lt => "<",
            FuzzBinOp::Le => "<=",
            FuzzBinOp::Gt => ">",
            FuzzBinOp::Ge => ">=",
            FuzzBinOp::Eq => "==",
            FuzzBinOp::Ne => "!=",
            FuzzBinOp::Shl => "<<",
            FuzzBinOp::Shr => ">>",
        }
    }
}

#[derive(Debug, Arbitrary)]
enum FuzzUnaryOp {
    Neg,
    Not,
}

impl FuzzUnaryOp {
    fn to_str(&self) -> &'static str {
        match self {
            FuzzUnaryOp::Neg => "-",
            FuzzUnaryOp::Not => "!",
        }
    }
}

#[derive(Debug, Arbitrary)]
enum FuzzIntrinsic {
    ThreadIdxX,
    ThreadIdxY,
    BlockIdxX,
    BlockDimX,
    GridDimX,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Min,
    Max,
    SyncThreads,
}

impl FuzzIntrinsic {
    fn to_str(&self) -> &'static str {
        match self {
            FuzzIntrinsic::ThreadIdxX => "thread_idx_x",
            FuzzIntrinsic::ThreadIdxY => "thread_idx_y",
            FuzzIntrinsic::BlockIdxX => "block_idx_x",
            FuzzIntrinsic::BlockDimX => "block_dim_x",
            FuzzIntrinsic::GridDimX => "grid_dim_x",
            FuzzIntrinsic::Sqrt => "sqrt",
            FuzzIntrinsic::Abs => "abs",
            FuzzIntrinsic::Sin => "sin",
            FuzzIntrinsic::Cos => "cos",
            FuzzIntrinsic::Min => "min",
            FuzzIntrinsic::Max => "max",
            FuzzIntrinsic::SyncThreads => "sync_threads",
        }
    }

    fn arg_count(&self) -> usize {
        match self {
            FuzzIntrinsic::ThreadIdxX
            | FuzzIntrinsic::ThreadIdxY
            | FuzzIntrinsic::BlockIdxX
            | FuzzIntrinsic::BlockDimX
            | FuzzIntrinsic::GridDimX
            | FuzzIntrinsic::SyncThreads => 0,
            FuzzIntrinsic::Sqrt | FuzzIntrinsic::Abs | FuzzIntrinsic::Sin | FuzzIntrinsic::Cos => 1,
            FuzzIntrinsic::Min | FuzzIntrinsic::Max => 2,
        }
    }
}

fn generate_expr(expr: &FuzzExpr, depth: usize) -> String {
    if depth > 10 {
        return "0".to_string(); // Prevent infinite recursion
    }

    match expr {
        FuzzExpr::Literal(lit) => match lit {
            FuzzLiteral::Int(v) => v.to_string(),
            FuzzLiteral::Float(v) => {
                let v = if v.is_nan() || v.is_infinite() { 0.0 } else { *v };
                format!("{:.6}", v)
            }
            FuzzLiteral::Bool(v) => v.to_string(),
        },
        FuzzExpr::Ident(id) => id.to_string(),
        FuzzExpr::Binary { op, left, right } => {
            format!(
                "({} {} {})",
                generate_expr(left, depth + 1),
                op.to_str(),
                generate_expr(right, depth + 1)
            )
        }
        FuzzExpr::Unary { op, operand } => {
            format!("({}{})", op.to_str(), generate_expr(operand, depth + 1))
        }
        FuzzExpr::Call { func, args } => {
            let arg_count = func.arg_count();
            let args_str: Vec<String> = args
                .iter()
                .take(arg_count)
                .map(|a| generate_expr(a, depth + 1))
                .collect();

            if args_str.len() < arg_count {
                // Not enough args, generate defaults
                let mut full_args = args_str;
                while full_args.len() < arg_count {
                    full_args.push("0".to_string());
                }
                format!("{}({})", func.to_str(), full_args.join(", "))
            } else {
                format!("{}({})", func.to_str(), args_str.join(", "))
            }
        }
        FuzzExpr::Index { array, index } => {
            format!("{}[{} as usize]", array.to_string(), generate_expr(index, depth + 1))
        }
        FuzzExpr::Cast { expr, ty } => {
            format!("({} as {})", generate_expr(expr, depth + 1), ty.to_rust_type())
        }
    }
}

fn generate_stmt(stmt: &FuzzStmt, depth: usize) -> String {
    if depth > 10 {
        return String::new();
    }

    match stmt {
        FuzzStmt::Let { name, value } => {
            format!("let {} = {};", name.to_string(), generate_expr(value, 0))
        }
        FuzzStmt::Assign { target, value } => {
            format!("{} = {};", target.to_string(), generate_expr(value, 0))
        }
        FuzzStmt::If { cond, then_body } => {
            let body: String = then_body
                .iter()
                .take(5)
                .map(|s| generate_stmt(s, depth + 1))
                .collect::<Vec<_>>()
                .join("\n");
            format!("if {} {{ {} }}", generate_expr(cond, 0), body)
        }
        FuzzStmt::IfElse {
            cond,
            then_body,
            else_body,
        } => {
            let then_str: String = then_body
                .iter()
                .take(5)
                .map(|s| generate_stmt(s, depth + 1))
                .collect::<Vec<_>>()
                .join("\n");
            let else_str: String = else_body
                .iter()
                .take(5)
                .map(|s| generate_stmt(s, depth + 1))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "if {} {{ {} }} else {{ {} }}",
                generate_expr(cond, 0),
                then_str,
                else_str
            )
        }
        FuzzStmt::For {
            var,
            start,
            end,
            body,
        } => {
            let body_str: String = body
                .iter()
                .take(5)
                .map(|s| generate_stmt(s, depth + 1))
                .collect::<Vec<_>>()
                .join("\n");
            let (s, e) = if start < end { (*start, *end) } else { (*end, *start) };
            let e = e.saturating_add(1).min(s.saturating_add(100));
            format!("for {} in {}..{} {{ {} }}", var.to_string(), s, e, body_str)
        }
        FuzzStmt::While { cond, body } => {
            let body_str: String = body
                .iter()
                .take(5)
                .map(|s| generate_stmt(s, depth + 1))
                .collect::<Vec<_>>()
                .join("\n");
            format!("while {} {{ {} }}", generate_expr(cond, 0), body_str)
        }
        FuzzStmt::Return(expr) => match expr {
            Some(e) => format!("return {};", generate_expr(e, 0)),
            None => "return;".to_string(),
        },
        FuzzStmt::Expr(e) => format!("{};", generate_expr(e, 0)),
    }
}

fn generate_kernel(kernel: &FuzzKernel) -> String {
    let mut code = String::new();

    // Function signature
    code.push_str(&format!("fn {}(", kernel.name.to_string()));

    let params: Vec<String> = kernel
        .params
        .iter()
        .take(8) // Limit params
        .map(|p| {
            let mut param = p.name.to_string();
            param.push_str(": ");
            if p.is_ref {
                param.push('&');
                if p.is_mut {
                    param.push_str("mut ");
                }
            }
            param.push_str(p.ty.to_rust_type());
            param
        })
        .collect();
    code.push_str(&params.join(", "));
    code.push_str(") {\n");

    // Body
    for stmt in kernel.body.iter().take(20) {
        code.push_str(&generate_stmt(stmt, 0));
        code.push('\n');
    }

    code.push_str("}\n");
    code
}

fuzz_target!(|kernel: FuzzKernel| {
    // Limit complexity
    if kernel.body.len() > 20 || kernel.params.len() > 8 {
        return;
    }

    // Generate Rust-like code
    let code = generate_kernel(&kernel);

    // Try to parse it with syn
    let parse_result = syn::parse_str::<syn::ItemFn>(&code);

    // If it parses, try to transpile
    if let Ok(item_fn) = parse_result {
        // Try global kernel transpilation - should not panic
        let _ = ringkernel_cuda_codegen::transpile_global_kernel(&item_fn);
    }
});
