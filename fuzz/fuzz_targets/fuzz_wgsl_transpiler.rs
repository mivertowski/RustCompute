//! Fuzz target for WGSL code generation.
//!
//! Tests the WGSL transpiler with random Rust-like input to find crashes
//! or panics in the code generation logic.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

/// Simplified kernel structure for WGSL fuzzing.
#[derive(Debug, Arbitrary)]
struct FuzzWgslKernel {
    name_suffix: u8,
    num_params: u8,
    body_ops: Vec<FuzzWgslOp>,
}

#[derive(Debug, Arbitrary)]
enum FuzzWgslOp {
    // GPU indexing
    GlobalInvocationId,
    LocalInvocationId,
    WorkgroupId,
    NumWorkgroups,

    // Math
    AddConst(i32),
    MulConst(i32),
    Sqrt,
    Abs,
    Min,
    Max,
    Clamp,

    // Control flow
    IfCheck,
    ForLoop { count: u8 },

    // Memory
    ArrayAccess { index: u8 },
    StorageAccess { index: u8 },
}

fn generate_wgsl_kernel(kernel: &FuzzWgslKernel) -> String {
    let name = format!("kernel_{}", kernel.name_suffix);
    let num_params = (kernel.num_params % 4) as usize + 1;

    let mut code = format!("fn {}(", name);

    // Generate parameters
    let params: Vec<String> = (0..num_params)
        .map(|i| {
            if i % 2 == 0 {
                format!("data{}: &mut [f32]", i)
            } else {
                format!("n{}: i32", i)
            }
        })
        .collect();
    code.push_str(&params.join(", "));
    code.push_str(") {\n");

    // Generate body
    code.push_str("    let idx = global_invocation_id_x() as i32;\n");

    for (i, op) in kernel.body_ops.iter().take(10).enumerate() {
        match op {
            FuzzWgslOp::GlobalInvocationId => {
                code.push_str(&format!(
                    "    let v{} = global_invocation_id_x();\n",
                    i
                ));
            }
            FuzzWgslOp::LocalInvocationId => {
                code.push_str(&format!(
                    "    let v{} = local_invocation_id_x();\n",
                    i
                ));
            }
            FuzzWgslOp::WorkgroupId => {
                code.push_str(&format!("    let v{} = workgroup_id_x();\n", i));
            }
            FuzzWgslOp::NumWorkgroups => {
                code.push_str(&format!("    let v{} = num_workgroups_x();\n", i));
            }
            FuzzWgslOp::AddConst(c) => {
                code.push_str(&format!("    let v{} = idx + {};\n", i, c));
            }
            FuzzWgslOp::MulConst(c) => {
                code.push_str(&format!("    let v{} = idx * {};\n", i, c));
            }
            FuzzWgslOp::Sqrt => {
                code.push_str(&format!("    let v{} = (idx as f32).sqrt();\n", i));
            }
            FuzzWgslOp::Abs => {
                code.push_str(&format!("    let v{} = idx.abs();\n", i));
            }
            FuzzWgslOp::Min => {
                code.push_str(&format!("    let v{} = idx.min(100);\n", i));
            }
            FuzzWgslOp::Max => {
                code.push_str(&format!("    let v{} = idx.max(0);\n", i));
            }
            FuzzWgslOp::Clamp => {
                code.push_str(&format!("    let v{} = idx.clamp(0, 100);\n", i));
            }
            FuzzWgslOp::IfCheck => {
                code.push_str(&format!(
                    "    if idx < 100 {{ let v{} = idx + 1; }}\n",
                    i
                ));
            }
            FuzzWgslOp::ForLoop { count } => {
                let count = (*count % 10) as i32 + 1;
                code.push_str(&format!(
                    "    for i in 0..{} {{ let v{} = i; }}\n",
                    count, i
                ));
            }
            FuzzWgslOp::ArrayAccess { index } => {
                let idx = *index as usize % num_params;
                if idx % 2 == 0 {
                    code.push_str(&format!(
                        "    if idx >= 0 && idx < 1000 {{ let v{} = data{}[idx as usize]; }}\n",
                        i, idx
                    ));
                }
            }
            FuzzWgslOp::StorageAccess { index } => {
                let idx = *index as usize % num_params;
                if idx % 2 == 0 {
                    code.push_str(&format!(
                        "    if idx >= 0 && idx < 1000 {{ data{}[idx as usize] = 1.0; }}\n",
                        idx
                    ));
                }
            }
        }
    }

    code.push_str("}\n");
    code
}

fuzz_target!(|kernel: FuzzWgslKernel| {
    // Limit complexity
    if kernel.body_ops.len() > 20 {
        return;
    }

    // Generate Rust-like code
    let code = generate_wgsl_kernel(&kernel);

    // Try to parse it with syn
    let parse_result = syn::parse_str::<syn::ItemFn>(&code);

    // If it parses, try to transpile to WGSL
    if let Ok(item_fn) = parse_result {
        // Try WGSL transpilation - should not panic
        let _ = ringkernel_wgpu_codegen::transpile_global_kernel(&item_fn);
    }
});
