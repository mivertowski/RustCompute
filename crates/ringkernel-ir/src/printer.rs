//! IR pretty printer.
//!
//! Produces human-readable text representation of IR modules.

use crate::{nodes::*, BlockId, IrModule, IrNode, IrType, Terminator, ValueId};
use std::fmt::Write;

/// IR pretty printer.
pub struct IrPrinter {
    indent: usize,
    output: String,
}

impl IrPrinter {
    /// Create a new printer.
    pub fn new() -> Self {
        Self {
            indent: 0,
            output: String::new(),
        }
    }

    /// Print a module.
    pub fn print(mut self, module: &IrModule) -> String {
        self.print_module(module);
        self.output
    }

    fn print_module(&mut self, module: &IrModule) {
        // Header
        writeln!(self.output, "; RingKernel IR Module: {}", module.name).unwrap();
        writeln!(
            self.output,
            "; Capabilities: {:?}",
            module.required_capabilities.flags()
        )
        .unwrap();
        writeln!(self.output).unwrap();

        // Parameters
        self.print_line("define kernel @");
        write!(self.output, "{}(", module.name).unwrap();
        for (i, param) in module.parameters.iter().enumerate() {
            if i > 0 {
                write!(self.output, ", ").unwrap();
            }
            write!(self.output, "{} %{}", param.ty, param.name).unwrap();
        }
        writeln!(self.output, ") {{").unwrap();

        self.indent += 1;

        // Print blocks in order (entry first)
        self.print_block(module, module.entry_block);
        for block_id in module.blocks.keys() {
            if *block_id != module.entry_block {
                self.print_block(module, *block_id);
            }
        }

        self.indent -= 1;
        self.print_line("}");
    }

    fn print_block(&mut self, module: &IrModule, block_id: BlockId) {
        let block = match module.blocks.get(&block_id) {
            Some(b) => b,
            None => return,
        };

        // Block label
        writeln!(self.output).unwrap();
        writeln!(self.output, "{}:", block.label).unwrap();

        // Instructions
        for inst in &block.instructions {
            self.print_instruction(module, inst.result, &inst.result_type, &inst.node);
        }

        // Terminator
        if let Some(term) = &block.terminator {
            self.print_terminator(term);
        }
    }

    fn print_instruction(
        &mut self,
        _module: &IrModule,
        result: ValueId,
        ty: &IrType,
        node: &IrNode,
    ) {
        let indent = "  ".repeat(self.indent);

        let node_str = match node {
            // Constants
            IrNode::Constant(c) => format!("{} = const {}", result, format_constant(c)),
            IrNode::Parameter(idx) => format!("{} = param {}", result, idx),
            IrNode::Undef => format!("{} = undef", result),

            // Binary ops
            IrNode::BinaryOp(op, lhs, rhs) => {
                format!("{} = {} {} {}, {}", result, op, ty, lhs, rhs)
            }

            // Unary ops
            IrNode::UnaryOp(op, val) => {
                format!("{} = {} {} {}", result, op, ty, val)
            }

            // Comparison
            IrNode::Compare(op, lhs, rhs) => {
                format!("{} = cmp {} {}, {}", result, op, lhs, rhs)
            }

            // Cast
            IrNode::Cast(kind, val, target_ty) => {
                format!("{} = cast {:?} {} to {}", result, kind, val, target_ty)
            }

            // Memory
            IrNode::Load(ptr) => format!("{} = load {}", result, ptr),
            IrNode::Store(ptr, val) => format!("store {}, {}", ptr, val),
            IrNode::GetElementPtr(ptr, indices) => {
                let indices_str: Vec<String> = indices.iter().map(|i| format!("{}", i)).collect();
                format!("{} = gep {}, [{}]", result, ptr, indices_str.join(", "))
            }
            IrNode::Alloca(ty) => format!("{} = alloca {}", result, ty),
            IrNode::SharedAlloc(ty, count) => {
                format!("{} = shared_alloc [{} x {}]", result, count, ty)
            }
            IrNode::ExtractField(val, idx) => {
                format!("{} = extractfield {}, {}", result, val, idx)
            }
            IrNode::InsertField(val, idx, new_val) => {
                format!("{} = insertfield {}, {}, {}", result, val, idx, new_val)
            }

            // GPU indexing
            IrNode::ThreadId(dim) => format!("{} = thread_id.{}", result, dim),
            IrNode::BlockId(dim) => format!("{} = block_id.{}", result, dim),
            IrNode::BlockDim(dim) => format!("{} = block_dim.{}", result, dim),
            IrNode::GridDim(dim) => format!("{} = grid_dim.{}", result, dim),
            IrNode::GlobalThreadId(dim) => format!("{} = global_thread_id.{}", result, dim),
            IrNode::WarpId => format!("{} = warp_id", result),
            IrNode::LaneId => format!("{} = lane_id", result),

            // Synchronization
            IrNode::Barrier => "barrier".to_string(),
            IrNode::MemoryFence(scope) => format!("fence {:?}", scope),
            IrNode::GridSync => "grid_sync".to_string(),

            // Atomics
            IrNode::Atomic(op, ptr, val) => {
                format!("{} = atomic_{:?} {}, {}", result, op, ptr, val)
            }
            IrNode::AtomicCas(ptr, expected, desired) => {
                format!("{} = atomic_cas {}, {}, {}", result, ptr, expected, desired)
            }

            // Warp ops
            IrNode::WarpVote(op, val) => format!("{} = warp_{:?} {}", result, op, val),
            IrNode::WarpShuffle(op, val, lane) => {
                format!("{} = warp_shuffle_{:?} {}, {}", result, op, val, lane)
            }
            IrNode::WarpReduce(op, val) => format!("{} = warp_reduce_{:?} {}", result, op, val),

            // Math
            IrNode::Math(op, args) => {
                let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
                format!("{} = {:?}({})", result, op, args_str.join(", "))
            }

            // Control flow
            IrNode::Select(cond, then_val, else_val) => {
                format!("{} = select {}, {}, {}", result, cond, then_val, else_val)
            }
            IrNode::Phi(entries) => {
                let entries_str: Vec<String> = entries
                    .iter()
                    .map(|(block, val)| format!("[{}, {}]", val, block))
                    .collect();
                format!("{} = phi {}", result, entries_str.join(", "))
            }

            // Messaging
            IrNode::K2HEnqueue(msg) => format!("k2h_enqueue {}", msg),
            IrNode::H2KDequeue => format!("{} = h2k_dequeue", result),
            IrNode::H2KIsEmpty => format!("{} = h2k_is_empty", result),
            IrNode::K2KSend(dest, msg) => format!("k2k_send {}, {}", dest, msg),
            IrNode::K2KRecv => format!("{} = k2k_recv", result),
            IrNode::K2KTryRecv => format!("{} = k2k_try_recv", result),

            // HLC
            IrNode::HlcNow => format!("{} = hlc_now", result),
            IrNode::HlcTick => format!("{} = hlc_tick", result),
            IrNode::HlcUpdate(ts) => format!("{} = hlc_update {}", result, ts),

            // Call
            IrNode::Call(name, args) => {
                let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
                format!("{} = call @{}({})", result, name, args_str.join(", "))
            }
        };

        writeln!(self.output, "{}{}", indent, node_str).unwrap();
    }

    fn print_terminator(&mut self, term: &Terminator) {
        let indent = "  ".repeat(self.indent);
        let term_str = match term {
            Terminator::Return(None) => "ret void".to_string(),
            Terminator::Return(Some(val)) => format!("ret {}", val),
            Terminator::Branch(target) => format!("br {}", target),
            Terminator::CondBranch(cond, then_block, else_block) => {
                format!("br {}, {}, {}", cond, then_block, else_block)
            }
            Terminator::Switch(val, default, cases) => {
                let cases_str: Vec<String> = cases
                    .iter()
                    .map(|(c, b)| format!("{} -> {}", format_constant(c), b))
                    .collect();
                format!(
                    "switch {}, default {}, [{}]",
                    val,
                    default,
                    cases_str.join(", ")
                )
            }
            Terminator::Unreachable => "unreachable".to_string(),
        };
        writeln!(self.output, "{}{}", indent, term_str).unwrap();
    }

    fn print_line(&mut self, text: &str) {
        let indent = "  ".repeat(self.indent);
        write!(self.output, "{}{}", indent, text).unwrap();
    }
}

impl Default for IrPrinter {
    fn default() -> Self {
        Self::new()
    }
}

fn format_constant(c: &ConstantValue) -> String {
    match c {
        ConstantValue::Bool(b) => format!("{}", b),
        ConstantValue::I32(v) => format!("{}i32", v),
        ConstantValue::I64(v) => format!("{}i64", v),
        ConstantValue::U32(v) => format!("{}u32", v),
        ConstantValue::U64(v) => format!("{}u64", v),
        ConstantValue::F32(v) => format!("{}f32", v),
        ConstantValue::F64(v) => format!("{}f64", v),
        ConstantValue::Null => "null".to_string(),
        ConstantValue::Array(elements) => {
            let elems: Vec<String> = elements.iter().map(format_constant).collect();
            format!("[{}]", elems.join(", "))
        }
        ConstantValue::Struct(fields) => {
            let fields_str: Vec<String> = fields.iter().map(format_constant).collect();
            format!("{{{}}}", fields_str.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dimension, IrBuilder};

    #[test]
    fn test_print_simple_kernel() {
        let mut builder = IrBuilder::new("saxpy");

        let _x = builder.parameter("x", IrType::ptr(IrType::F32));
        let _y = builder.parameter("y", IrType::ptr(IrType::F32));
        let _a = builder.parameter("a", IrType::F32);

        let idx = builder.thread_id(Dimension::X);
        let _ = idx; // Would be used for indexing

        builder.ret();

        let module = builder.build();
        let output = module.pretty_print();

        assert!(output.contains("saxpy"));
        assert!(output.contains("thread_id.x"));
        assert!(output.contains("ret void"));
    }

    #[test]
    fn test_print_with_arithmetic() {
        let mut builder = IrBuilder::new("test");

        let a = builder.const_i32(10);
        let b = builder.const_i32(20);
        let c = builder.add(a, b);
        let _ = c;

        builder.ret();

        let module = builder.build();
        let output = module.pretty_print();

        // Constants are stored as values, not printed in blocks
        // The add instruction references them by ValueId
        assert!(output.contains("add"));
        assert!(output.contains("i32")); // Type annotation in add
    }

    #[test]
    fn test_print_with_control_flow() {
        let mut builder = IrBuilder::new("test");

        let cond = builder.const_bool(true);
        let then_block = builder.create_block("then");
        let else_block = builder.create_block("else");

        builder.cond_branch(cond, then_block, else_block);

        builder.switch_to_block(then_block);
        builder.ret();

        builder.switch_to_block(else_block);
        builder.ret();

        let module = builder.build();
        let output = module.pretty_print();

        assert!(output.contains("then:"));
        assert!(output.contains("else:"));
        assert!(output.contains("br"));
    }
}
