//! Handler function integration for ring kernel transpilation.
//!
//! This module provides utilities for parsing handler function signatures,
//! extracting message/response types, and generating the code that binds
//! handlers to the ring kernel message loop.
//!
//! # Handler Signature Patterns
//!
//! Ring kernel handlers follow specific patterns:
//!
//! ```ignore
//! // Pattern 1: Message in, response out
//! fn handle(ctx: &RingContext, msg: &MyMessage) -> MyResponse { ... }
//!
//! // Pattern 2: Message in, no response (fire-and-forget)
//! fn handle(ctx: &RingContext, msg: &MyMessage) { ... }
//!
//! // Pattern 3: Simple value processing
//! fn process(value: f32) -> f32 { ... }
//! ```
//!
//! # Generated Code
//!
//! The handler body is embedded within the message loop with proper
//! message deserialization and response serialization:
//!
//! ```cuda
//! // Message pointer from buffer
//! MyMessage* msg = (MyMessage*)&input_buffer[msg_idx * MSG_SIZE];
//!
//! // === Handler body (transpiled) ===
//! float result = msg->value * 2.0f;
//! MyResponse response;
//! response.value = result;
//! // ================================
//!
//! // Enqueue response
//! memcpy(&output_buffer[out_idx * RESP_SIZE], &response, sizeof(MyResponse));
//! ```

use crate::types::{is_ring_context_type, CudaType, TypeMapper};
use crate::Result;
use std::fmt::Write;
use syn::{FnArg, ItemFn, Pat, ReturnType, Type};

/// Information about a handler function parameter.
#[derive(Debug, Clone)]
pub struct HandlerParam {
    /// Parameter name.
    pub name: String,
    /// Parameter type (Rust).
    pub rust_type: String,
    /// Parameter type (CUDA).
    pub cuda_type: String,
    /// Kind of parameter.
    pub kind: HandlerParamKind,
}

/// Kinds of handler parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandlerParamKind {
    /// RingContext reference (removed in transpilation).
    Context,
    /// Message reference (deserialized from input buffer).
    Message,
    /// Mutable message reference.
    MessageMut,
    /// Regular value parameter.
    Value,
    /// Slice parameter (pointer in CUDA).
    Slice,
    /// Mutable slice parameter.
    SliceMut,
}

/// Parsed handler function signature.
#[derive(Debug, Clone)]
pub struct HandlerSignature {
    /// Handler function name.
    pub name: String,
    /// All parameters.
    pub params: Vec<HandlerParam>,
    /// Return type (if any).
    pub return_type: Option<HandlerReturnType>,
    /// Whether the handler has a RingContext parameter.
    pub has_context: bool,
    /// The message parameter (if any).
    pub message_param: Option<HandlerParam>,
}

/// Handler return type information.
#[derive(Debug, Clone)]
pub struct HandlerReturnType {
    /// Rust type name.
    pub rust_type: String,
    /// CUDA type string.
    pub cuda_type: String,
    /// Whether this is a struct type (vs primitive).
    pub is_struct: bool,
}

impl HandlerSignature {
    /// Parse a handler function signature.
    pub fn parse(func: &ItemFn, type_mapper: &TypeMapper) -> Result<Self> {
        let name = func.sig.ident.to_string();
        let mut params = Vec::new();
        let mut has_context = false;
        let mut message_param = None;

        for param in &func.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                let param_name = match pat_type.pat.as_ref() {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    _ => continue,
                };

                let ty = &pat_type.ty;
                let rust_type = quote::quote!(#ty).to_string();
                let kind = Self::classify_param(&param_name, &pat_type.ty);

                if kind == HandlerParamKind::Context {
                    has_context = true;
                    continue; // Skip context params in output
                }

                let cuda_type = match type_mapper.map_type(&pat_type.ty) {
                    Ok(ct) => ct.to_cuda_string(),
                    Err(_) => "void*".to_string(), // Fallback for unknown types
                };

                let param = HandlerParam {
                    name: param_name,
                    rust_type,
                    cuda_type,
                    kind,
                };

                if kind == HandlerParamKind::Message || kind == HandlerParamKind::MessageMut {
                    message_param = Some(param.clone());
                }

                params.push(param);
            }
        }

        let return_type = Self::parse_return_type(&func.sig.output, type_mapper)?;

        Ok(Self {
            name,
            params,
            return_type,
            has_context,
            message_param,
        })
    }

    /// Classify a parameter based on its name and type.
    fn classify_param(name: &str, ty: &Type) -> HandlerParamKind {
        // Check for RingContext
        if is_ring_context_type(ty) {
            return HandlerParamKind::Context;
        }

        // Check name patterns
        let name_lower = name.to_lowercase();
        if name_lower == "ctx" || name_lower == "context" {
            return HandlerParamKind::Context;
        }

        // Check for references
        if let Type::Reference(reference) = ty {
            let is_mut = reference.mutability.is_some();

            // Check for slice
            if let Type::Slice(_) = reference.elem.as_ref() {
                return if is_mut {
                    HandlerParamKind::SliceMut
                } else {
                    HandlerParamKind::Slice
                };
            }

            // Message-like parameters (references to structs)
            if name_lower == "msg" || name_lower == "message" || name_lower.starts_with("msg_") {
                return if is_mut {
                    HandlerParamKind::MessageMut
                } else {
                    HandlerParamKind::Message
                };
            }

            // Generic reference - treat as message if it's a struct
            if let Type::Path(path) = reference.elem.as_ref() {
                let type_name = path
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();

                // Heuristic: if type name ends with "Message" or "Request", treat as message
                if type_name.ends_with("Message") || type_name.ends_with("Request") {
                    return if is_mut {
                        HandlerParamKind::MessageMut
                    } else {
                        HandlerParamKind::Message
                    };
                }
            }
        }

        HandlerParamKind::Value
    }

    /// Parse the return type.
    fn parse_return_type(
        output: &ReturnType,
        type_mapper: &TypeMapper,
    ) -> Result<Option<HandlerReturnType>> {
        match output {
            ReturnType::Default => Ok(None),
            ReturnType::Type(_, ty) => {
                // Check for unit type
                if let Type::Tuple(tuple) = ty.as_ref() {
                    if tuple.elems.is_empty() {
                        return Ok(None);
                    }
                }

                let rust_type = quote::quote!(#ty).to_string();
                let cuda_type = type_mapper
                    .map_type(ty)
                    .map(|ct| ct.to_cuda_string())
                    .unwrap_or_else(|_| rust_type.clone());

                let is_struct = matches!(type_mapper.map_type(ty), Ok(CudaType::Struct(_)));

                Ok(Some(HandlerReturnType {
                    rust_type,
                    cuda_type,
                    is_struct,
                }))
            }
        }
    }

    /// Check if the handler produces a response.
    pub fn has_response(&self) -> bool {
        self.return_type.is_some()
    }

    /// Get non-context, non-message parameters (additional kernel params).
    pub fn extra_params(&self) -> Vec<&HandlerParam> {
        self.params
            .iter()
            .filter(|p| {
                p.kind != HandlerParamKind::Context
                    && p.kind != HandlerParamKind::Message
                    && p.kind != HandlerParamKind::MessageMut
            })
            .collect()
    }
}

/// Configuration for handler code generation.
#[derive(Debug, Clone)]
pub struct HandlerCodegenConfig {
    /// Name for the message pointer variable.
    pub message_var: String,
    /// Name for the response variable.
    pub response_var: String,
    /// Indent string for generated code.
    pub indent: String,
    /// Whether to generate message deserialization.
    pub generate_deser: bool,
    /// Whether to generate response serialization.
    pub generate_ser: bool,
}

impl Default for HandlerCodegenConfig {
    fn default() -> Self {
        Self {
            message_var: "msg".to_string(),
            response_var: "response".to_string(),
            indent: "        ".to_string(), // 2 levels (function + loop)
            generate_deser: true,
            generate_ser: true,
        }
    }
}

/// Generate message deserialization code.
///
/// This generates code to cast the input buffer pointer to the message type.
/// When envelope format is used, msg_ptr points to the payload (after header).
pub fn generate_message_deser(message_type: &str, config: &HandlerCodegenConfig) -> String {
    let mut code = String::new();
    let indent = &config.indent;

    writeln!(code, "{}// Deserialize message from buffer", indent).unwrap();
    writeln!(
        code,
        "{}// msg_ptr points to payload data (after MessageHeader when using envelopes)",
        indent
    )
    .unwrap();
    writeln!(
        code,
        "{}{}* {} = ({}*)msg_ptr;",
        indent, message_type, config.message_var, message_type
    )
    .unwrap();

    code
}

/// Generate message deserialization code for envelope format with header access.
///
/// This provides access to both the message header and typed payload.
pub fn generate_envelope_message_deser(
    message_type: &str,
    config: &HandlerCodegenConfig,
) -> String {
    let mut code = String::new();
    let indent = &config.indent;

    writeln!(code, "{}// Message envelope deserialization", indent).unwrap();
    writeln!(
        code,
        "{}// msg_header provides: message_id, correlation_id, source_kernel, timestamp, etc.",
        indent
    )
    .unwrap();
    writeln!(
        code,
        "{}{}* {} = ({}*)msg_ptr;  // Typed payload",
        indent, message_type, config.message_var, message_type
    )
    .unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "{}// Access header fields:",
        indent
    )
    .unwrap();
    writeln!(
        code,
        "{}// - msg_header->message_id     (unique message ID)",
        indent
    )
    .unwrap();
    writeln!(
        code,
        "{}// - msg_header->correlation_id (for request-response matching)",
        indent
    )
    .unwrap();
    writeln!(
        code,
        "{}// - msg_header->source_kernel  (sender kernel ID, 0 = host)",
        indent
    )
    .unwrap();
    writeln!(
        code,
        "{}// - msg_header->timestamp      (HLC timestamp of send)",
        indent
    )
    .unwrap();

    code
}

/// Generate response serialization code (legacy raw format).
///
/// This generates code to copy the response to the output buffer.
pub fn generate_response_ser(response_type: &str, config: &HandlerCodegenConfig) -> String {
    let mut code = String::new();
    let indent = &config.indent;

    writeln!(code).unwrap();
    writeln!(code, "{}// Serialize response to output buffer (raw format)", indent).unwrap();
    writeln!(
        code,
        "{}unsigned long long _out_idx = atomicAdd(&control->output_head, 1) & control->output_mask;",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}memcpy(&output_buffer[_out_idx * RESP_SIZE], &{}, sizeof({}));",
        indent, config.response_var, response_type
    )
    .unwrap();

    code
}

/// Generate response serialization code for envelope format.
///
/// This creates a complete response envelope with header and payload,
/// properly routing the response back to the sender.
pub fn generate_envelope_response_ser(
    response_type: &str,
    config: &HandlerCodegenConfig,
) -> String {
    let mut code = String::new();
    let indent = &config.indent;

    writeln!(code).unwrap();
    writeln!(code, "{}// Serialize response envelope to output buffer", indent).unwrap();
    writeln!(
        code,
        "{}unsigned long long _out_idx = atomicAdd(&control->output_head, 1) & control->output_mask;",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}unsigned char* resp_envelope = &output_buffer[_out_idx * RESP_SIZE];",
        indent
    ).unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "{}// Build response header from request header",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}MessageHeader* resp_header = (MessageHeader*)resp_envelope;",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}message_create_response_header(",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}    resp_header,",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}    msg_header,              // Request header (for correlation)",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}    KERNEL_ID,               // This kernel's ID",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}    sizeof({}),   // Payload size",
        indent, response_type
    ).unwrap();
    writeln!(
        code,
        "{}    hlc_physical,            // Current HLC",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}    hlc_logical,",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}    HLC_NODE_ID",
        indent
    ).unwrap();
    writeln!(
        code,
        "{});",
        indent
    ).unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "{}// Copy response payload after header",
        indent
    ).unwrap();
    writeln!(
        code,
        "{}memcpy(resp_envelope + MESSAGE_HEADER_SIZE, &{}, sizeof({}));",
        indent, config.response_var, response_type
    ).unwrap();
    writeln!(code).unwrap();
    writeln!(
        code,
        "{}__threadfence();  // Ensure write is visible",
        indent
    ).unwrap();

    code
}

/// Generate a CUDA struct definition from field information.
pub fn generate_cuda_struct(
    name: &str,
    fields: &[(String, String)], // (field_name, cuda_type)
) -> String {
    let mut code = String::new();

    writeln!(code, "struct {} {{", name).unwrap();
    for (field_name, cuda_type) in fields {
        writeln!(code, "    {} {};", cuda_type, field_name).unwrap();
    }
    writeln!(code, "}};").unwrap();

    code
}

/// Message type information for code generation.
#[derive(Debug, Clone)]
pub struct MessageTypeInfo {
    /// Type name.
    pub name: String,
    /// Size in bytes.
    pub size: usize,
    /// Fields (name, cuda_type).
    pub fields: Vec<(String, String)>,
}

impl MessageTypeInfo {
    /// Create a simple message type with a single value field.
    pub fn simple(name: &str, value_type: &str) -> Self {
        Self {
            name: name.to_string(),
            size: Self::type_size(value_type),
            fields: vec![("value".to_string(), value_type.to_string())],
        }
    }

    /// Get the size of a CUDA type in bytes.
    fn type_size(cuda_type: &str) -> usize {
        match cuda_type {
            "float" => 4,
            "double" => 8,
            "int" => 4,
            "unsigned int" => 4,
            "long long" | "unsigned long long" => 8,
            "short" | "unsigned short" => 2,
            "char" | "unsigned char" => 1,
            _ => 8, // Default assumption
        }
    }

    /// Generate CUDA struct definition.
    pub fn to_cuda_struct(&self) -> String {
        generate_cuda_struct(&self.name, &self.fields)
    }
}

/// Registry of message types for a kernel.
#[derive(Debug, Clone, Default)]
pub struct MessageTypeRegistry {
    /// Registered message types.
    pub messages: Vec<MessageTypeInfo>,
    /// Registered response types.
    pub responses: Vec<MessageTypeInfo>,
}

impl MessageTypeRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a message type.
    pub fn register_message(&mut self, info: MessageTypeInfo) {
        self.messages.push(info);
    }

    /// Register a response type.
    pub fn register_response(&mut self, info: MessageTypeInfo) {
        self.responses.push(info);
    }

    /// Generate all struct definitions.
    pub fn generate_structs(&self) -> String {
        let mut code = String::new();

        for msg in &self.messages {
            code.push_str(&msg.to_cuda_struct());
            code.push('\n');
        }

        for resp in &self.responses {
            code.push_str(&resp.to_cuda_struct());
            code.push('\n');
        }

        code
    }
}

/// RingContext method mappings to CUDA intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMethod {
    /// ctx.thread_id() -> threadIdx.x
    ThreadId,
    /// ctx.block_id() -> blockIdx.x
    BlockId,
    /// ctx.global_thread_id() -> blockIdx.x * blockDim.x + threadIdx.x
    GlobalThreadId,
    /// ctx.sync_threads() -> __syncthreads()
    SyncThreads,
    /// ctx.now() -> HLC timestamp
    Now,
    /// ctx.atomic_add(ptr, val) -> atomicAdd(ptr, val)
    AtomicAdd,
    /// ctx.warp_shuffle(val, lane) -> __shfl_sync(0xFFFFFFFF, val, lane)
    WarpShuffle,
    /// ctx.lane_id() -> threadIdx.x % 32
    LaneId,
    /// ctx.warp_id() -> threadIdx.x / 32
    WarpId,
}

impl ContextMethod {
    /// Parse a method name to context method.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "thread_id" | "thread_idx" => Some(Self::ThreadId),
            "block_id" | "block_idx" => Some(Self::BlockId),
            "global_thread_id" | "global_id" => Some(Self::GlobalThreadId),
            "sync_threads" | "synchronize" => Some(Self::SyncThreads),
            "now" | "timestamp" => Some(Self::Now),
            "atomic_add" => Some(Self::AtomicAdd),
            "warp_shuffle" | "shuffle" => Some(Self::WarpShuffle),
            "lane_id" => Some(Self::LaneId),
            "warp_id" => Some(Self::WarpId),
            _ => None,
        }
    }

    /// Generate CUDA code for this context method.
    pub fn to_cuda(&self, args: &[String]) -> String {
        match self {
            Self::ThreadId => "threadIdx.x".to_string(),
            Self::BlockId => "blockIdx.x".to_string(),
            Self::GlobalThreadId => "(blockIdx.x * blockDim.x + threadIdx.x)".to_string(),
            Self::SyncThreads => "__syncthreads()".to_string(),
            Self::Now => "(hlc_physical << 32) | (hlc_logical & 0xFFFFFFFF)".to_string(),
            Self::AtomicAdd => {
                if args.len() >= 2 {
                    format!("atomicAdd({}, {})", args[0], args[1])
                } else {
                    "/* atomic_add requires ptr and val */".to_string()
                }
            }
            Self::WarpShuffle => {
                if args.len() >= 2 {
                    format!("__shfl_sync(0xFFFFFFFF, {}, {})", args[0], args[1])
                } else {
                    "/* warp_shuffle requires val and lane */".to_string()
                }
            }
            Self::LaneId => "(threadIdx.x % 32)".to_string(),
            Self::WarpId => "(threadIdx.x / 32)".to_string(),
        }
    }

    /// Check if this method is a statement (vs expression).
    pub fn is_statement(&self) -> bool {
        matches!(self, Self::SyncThreads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_handler_signature_simple() {
        let func: ItemFn = parse_quote! {
            fn process(value: f32) -> f32 {
                value * 2.0
            }
        };

        let mapper = TypeMapper::new();
        let sig = HandlerSignature::parse(&func, &mapper).unwrap();

        assert_eq!(sig.name, "process");
        assert_eq!(sig.params.len(), 1);
        assert_eq!(sig.params[0].name, "value");
        assert_eq!(sig.params[0].kind, HandlerParamKind::Value);
        assert!(sig.has_response());
        assert!(!sig.has_context);
    }

    #[test]
    fn test_handler_signature_with_context() {
        let func: ItemFn = parse_quote! {
            fn handle(ctx: &RingContext, msg: &Message) -> Response {
                Response { value: msg.value * 2.0 }
            }
        };

        let mapper = TypeMapper::new();
        let sig = HandlerSignature::parse(&func, &mapper).unwrap();

        assert_eq!(sig.name, "handle");
        assert!(sig.has_context);
        assert!(sig.message_param.is_some());
        assert_eq!(sig.message_param.as_ref().unwrap().name, "msg");
        assert!(sig.has_response());
    }

    #[test]
    fn test_handler_signature_no_response() {
        let func: ItemFn = parse_quote! {
            fn fire_and_forget(msg: &Event) {
                process(msg);
            }
        };

        let mapper = TypeMapper::new();
        let sig = HandlerSignature::parse(&func, &mapper).unwrap();

        assert!(!sig.has_response());
    }

    #[test]
    fn test_message_deser_generation() {
        let config = HandlerCodegenConfig::default();
        let code = generate_message_deser("MyMessage", &config);

        assert!(code.contains("MyMessage* msg"));
        assert!(code.contains("(MyMessage*)msg_ptr"));
    }

    #[test]
    fn test_response_ser_generation() {
        let config = HandlerCodegenConfig::default();
        let code = generate_response_ser("MyResponse", &config);

        assert!(code.contains("memcpy"));
        assert!(code.contains("output_buffer"));
        assert!(code.contains("RESP_SIZE"));
    }

    #[test]
    fn test_cuda_struct_generation() {
        let fields = vec![
            ("id".to_string(), "unsigned long long".to_string()),
            ("value".to_string(), "float".to_string()),
        ];

        let code = generate_cuda_struct("MyMessage", &fields);

        assert!(code.contains("struct MyMessage"));
        assert!(code.contains("unsigned long long id"));
        assert!(code.contains("float value"));
    }

    #[test]
    fn test_message_type_info() {
        let info = MessageTypeInfo::simple("FloatMsg", "float");
        assert_eq!(info.name, "FloatMsg");
        assert_eq!(info.size, 4);
        assert_eq!(info.fields.len(), 1);

        let cuda = info.to_cuda_struct();
        assert!(cuda.contains("struct FloatMsg"));
        assert!(cuda.contains("float value"));
    }

    #[test]
    fn test_context_method_lookup() {
        assert_eq!(
            ContextMethod::from_name("thread_id"),
            Some(ContextMethod::ThreadId)
        );
        assert_eq!(
            ContextMethod::from_name("sync_threads"),
            Some(ContextMethod::SyncThreads)
        );
        assert_eq!(
            ContextMethod::from_name("global_thread_id"),
            Some(ContextMethod::GlobalThreadId)
        );
        assert_eq!(ContextMethod::from_name("unknown"), None);
    }

    #[test]
    fn test_context_method_cuda_output() {
        assert_eq!(ContextMethod::ThreadId.to_cuda(&[]), "threadIdx.x");
        assert_eq!(ContextMethod::SyncThreads.to_cuda(&[]), "__syncthreads()");
        assert_eq!(
            ContextMethod::GlobalThreadId.to_cuda(&[]),
            "(blockIdx.x * blockDim.x + threadIdx.x)"
        );

        let args = vec!["ptr".to_string(), "1".to_string()];
        assert_eq!(ContextMethod::AtomicAdd.to_cuda(&args), "atomicAdd(ptr, 1)");
    }

    #[test]
    fn test_message_type_registry() {
        let mut registry = MessageTypeRegistry::new();

        registry.register_message(MessageTypeInfo {
            name: "Request".to_string(),
            size: 8,
            fields: vec![("value".to_string(), "float".to_string())],
        });

        registry.register_response(MessageTypeInfo {
            name: "Response".to_string(),
            size: 8,
            fields: vec![("result".to_string(), "float".to_string())],
        });

        let structs = registry.generate_structs();
        assert!(structs.contains("struct Request"));
        assert!(structs.contains("struct Response"));
    }

    #[test]
    fn test_handler_param_kind_classification() {
        let ctx_ty: Type = parse_quote!(&RingContext);
        assert_eq!(
            HandlerSignature::classify_param("ctx", &ctx_ty),
            HandlerParamKind::Context
        );

        let msg_ty: Type = parse_quote!(&MyMessage);
        assert_eq!(
            HandlerSignature::classify_param("msg", &msg_ty),
            HandlerParamKind::Message
        );

        let slice_ty: Type = parse_quote!(&[f32]);
        assert_eq!(
            HandlerSignature::classify_param("data", &slice_ty),
            HandlerParamKind::Slice
        );

        let value_ty: Type = parse_quote!(f32);
        assert_eq!(
            HandlerSignature::classify_param("value", &value_ty),
            HandlerParamKind::Value
        );
    }
}
