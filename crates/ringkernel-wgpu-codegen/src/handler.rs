//! Handler function parsing and codegen configuration.
//!
//! Parses Rust handler functions and prepares them for WGSL transpilation.

/// Configuration for handler code generation.
#[derive(Debug, Clone, Default)]
pub struct HandlerCodegenConfig {
    /// Whether to inline context method calls.
    pub inline_context_methods: bool,
    /// Whether to generate bounds checking.
    pub bounds_checking: bool,
}

impl HandlerCodegenConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self {
            inline_context_methods: true,
            bounds_checking: true,
        }
    }

    /// Disable context method inlining.
    pub fn without_inlining(mut self) -> Self {
        self.inline_context_methods = false;
        self
    }

    /// Disable bounds checking.
    pub fn without_bounds_checking(mut self) -> Self {
        self.bounds_checking = false;
        self
    }
}

/// Parsed handler signature.
#[derive(Debug, Clone)]
pub struct HandlerSignature {
    /// Function name.
    pub name: String,
    /// Parameters.
    pub params: Vec<HandlerParam>,
    /// Return type.
    pub return_type: HandlerReturnType,
    /// Whether the handler takes a context parameter.
    pub has_context: bool,
    /// Index of the message parameter (if any).
    pub message_param: Option<usize>,
}

/// Handler parameter description.
#[derive(Debug, Clone)]
pub struct HandlerParam {
    /// Parameter name.
    pub name: String,
    /// Parameter kind.
    pub kind: HandlerParamKind,
    /// WGSL type string.
    pub wgsl_type: String,
}

/// Kind of handler parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerParamKind {
    /// RingContext parameter.
    Context,
    /// Message parameter.
    Message,
    /// Buffer parameter (slice).
    Buffer { mutable: bool },
    /// Scalar value.
    Scalar,
}

/// Handler return type.
#[derive(Debug, Clone)]
pub enum HandlerReturnType {
    /// No return value.
    Unit,
    /// Returns a value type.
    Value(String),
    /// Returns a message type.
    Message(String),
}

/// WGSL context method that can be inlined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslContextMethod {
    /// Thread ID within workgroup.
    LocalId,
    /// Global thread ID.
    GlobalId,
    /// Workgroup ID.
    WorkgroupId,
    /// Workgroup barrier.
    WorkgroupBarrier,
    /// Storage barrier.
    StorageBarrier,
    /// Atomic add.
    AtomicAdd,
    /// Atomic load.
    AtomicLoad,
    /// Atomic store.
    AtomicStore,
}

impl WgslContextMethod {
    /// Get the WGSL code for this context method.
    pub fn to_wgsl(&self) -> &'static str {
        match self {
            WgslContextMethod::LocalId => "local_invocation_id.x",
            WgslContextMethod::GlobalId => "global_invocation_id.x",
            WgslContextMethod::WorkgroupId => "workgroup_id.x",
            WgslContextMethod::WorkgroupBarrier => "workgroupBarrier()",
            WgslContextMethod::StorageBarrier => "storageBarrier()",
            WgslContextMethod::AtomicAdd => "atomicAdd",
            WgslContextMethod::AtomicLoad => "atomicLoad",
            WgslContextMethod::AtomicStore => "atomicStore",
        }
    }

    /// Look up a context method by name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "thread_id" | "local_id" => Some(WgslContextMethod::LocalId),
            "global_thread_id" | "global_id" => Some(WgslContextMethod::GlobalId),
            "workgroup_id" | "block_id" => Some(WgslContextMethod::WorkgroupId),
            "sync_threads" | "barrier" => Some(WgslContextMethod::WorkgroupBarrier),
            "thread_fence" | "storage_fence" => Some(WgslContextMethod::StorageBarrier),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_codegen_config() {
        let config = HandlerCodegenConfig::new()
            .without_inlining()
            .without_bounds_checking();

        assert!(!config.inline_context_methods);
        assert!(!config.bounds_checking);
    }

    #[test]
    fn test_context_method_lookup() {
        assert_eq!(
            WgslContextMethod::from_name("thread_id"),
            Some(WgslContextMethod::LocalId)
        );
        assert_eq!(
            WgslContextMethod::from_name("sync_threads"),
            Some(WgslContextMethod::WorkgroupBarrier)
        );
        assert_eq!(WgslContextMethod::from_name("unknown"), None);
    }

    #[test]
    fn test_context_method_wgsl() {
        assert_eq!(
            WgslContextMethod::LocalId.to_wgsl(),
            "local_invocation_id.x"
        );
        assert_eq!(
            WgslContextMethod::WorkgroupBarrier.to_wgsl(),
            "workgroupBarrier()"
        );
    }
}
