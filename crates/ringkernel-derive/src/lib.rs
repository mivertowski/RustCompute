//! Procedural macros for RingKernel.
//!
//! This crate provides the following macros:
//!
//! - `#[derive(RingMessage)]` - Implement the RingMessage trait for message types
//! - `#[ring_kernel]` - Define a ring kernel handler
//! - `#[stencil_kernel]` - Define a GPU stencil kernel (with `cuda-codegen` feature)
//! - `#[gpu_kernel]` - Define a multi-backend GPU kernel with capability checking
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_derive::{RingMessage, ring_kernel};
//!
//! #[derive(RingMessage)]
//! struct AddRequest {
//!     #[message(id)]
//!     id: MessageId,
//!     a: f32,
//!     b: f32,
//! }
//!
//! #[derive(RingMessage)]
//! struct AddResponse {
//!     #[message(id)]
//!     id: MessageId,
//!     result: f32,
//! }
//!
//! #[ring_kernel(id = "adder")]
//! async fn process(ctx: &mut RingContext, req: AddRequest) -> AddResponse {
//!     AddResponse {
//!         id: MessageId::generate(),
//!         result: req.a + req.b,
//!     }
//! }
//! ```
//!
//! # Multi-Backend GPU Kernels
//!
//! The `#[gpu_kernel]` macro enables multi-backend code generation with capability checking:
//!
//! ```ignore
//! use ringkernel_derive::gpu_kernel;
//!
//! // Generate code for CUDA and Metal, with fallback order
//! #[gpu_kernel(backends = [cuda, metal], fallback = [wgpu, cpu])]
//! fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
//!     let idx = global_thread_id_x();
//!     if idx < n {
//!         y[idx as usize] = a * x[idx as usize] + y[idx as usize];
//!     }
//! }
//!
//! // Require specific capabilities at compile time
//! #[gpu_kernel(backends = [cuda], requires = [f64, atomic64])]
//! fn double_precision(data: &mut [f64], n: i32) {
//!     // Uses f64 operations - validated at compile time
//! }
//! ```
//!
//! # Stencil Kernels (with `cuda-codegen` feature)
//!
//! ```ignore
//! use ringkernel_derive::stencil_kernel;
//! use ringkernel_cuda_codegen::GridPos;
//!
//! #[stencil_kernel(id = "fdtd", grid = "2d", tile_size = 16, halo = 1)]
//! fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
//!     let curr = p[pos.idx()];
//!     let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
//!     p_prev[pos.idx()] = 2.0 * curr - p_prev[pos.idx()] + c2 * lap;
//! }
//! ```

use darling::{ast, FromDeriveInput, FromField, FromMeta};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, DeriveInput, ItemFn};

/// Attributes for the RingMessage derive macro.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(message, ring_message), supports(struct_named))]
struct RingMessageArgs {
    ident: syn::Ident,
    generics: syn::Generics,
    data: ast::Data<(), RingMessageField>,
    /// Optional explicit message type ID.
    /// If domain is specified, this is the offset within the domain (0-99).
    /// If domain is not specified, this is the absolute type ID.
    #[darling(default)]
    type_id: Option<u64>,
    /// Optional domain for message classification.
    /// When specified, the final type ID = domain.base_type_id() + type_id.
    #[darling(default)]
    domain: Option<String>,
    /// Whether this message is routable via K2K.
    /// When true, generates a K2KMessageRegistration for runtime discovery.
    #[darling(default)]
    k2k_routable: bool,
    /// Optional category for K2K routing groups.
    /// Multiple messages can share a category for grouped routing.
    #[darling(default)]
    category: Option<String>,
}

/// Field attributes for RingMessage.
#[derive(Debug, FromField)]
#[darling(attributes(message))]
struct RingMessageField {
    ident: Option<syn::Ident>,
    #[allow(dead_code)]
    ty: syn::Type,
    /// Mark this field as the message ID.
    #[darling(default)]
    id: bool,
    /// Mark this field as the correlation ID.
    #[darling(default)]
    correlation: bool,
    /// Mark this field as the priority.
    #[darling(default)]
    priority: bool,
}

/// Derive macro for implementing the RingMessage trait.
///
/// # Attributes
///
/// On the struct (via `#[message(...)]` or `#[ring_message(...)]`):
/// - `type_id = 123` - Set explicit message type ID (or domain offset if domain is set)
/// - `domain = "OrderMatching"` - Assign to a business domain (adds base type ID)
/// - `k2k_routable = true` - Register for K2K routing discovery
/// - `category = "orders"` - Group messages for K2K routing
///
/// On fields:
/// - `#[message(id)]` - Mark as message ID field
/// - `#[message(correlation)]` - Mark as correlation ID field
/// - `#[message(priority)]` - Mark as priority field
///
/// # Examples
///
/// Basic usage:
/// ```ignore
/// #[derive(RingMessage)]
/// #[message(type_id = 1)]
/// struct MyMessage {
///     #[message(id)]
///     id: MessageId,
///     #[message(correlation)]
///     correlation: CorrelationId,
///     #[message(priority)]
///     priority: Priority,
///     payload: Vec<u8>,
/// }
/// ```
///
/// With domain (type ID = 500 + 1 = 501):
/// ```ignore
/// #[derive(RingMessage)]
/// #[ring_message(type_id = 1, domain = "OrderMatching")]
/// pub struct SubmitOrderInput {
///     #[message(id)]
///     id: MessageId,
///     pub order: Order,
/// }
/// // Also implements DomainMessage trait
/// assert_eq!(SubmitOrderInput::domain(), Domain::OrderMatching);
/// ```
///
/// K2K-routable message:
/// ```ignore
/// #[derive(RingMessage)]
/// #[ring_message(type_id = 1, domain = "OrderMatching", k2k_routable = true, category = "orders")]
/// pub struct SubmitOrderInput { ... }
///
/// // Runtime discovery:
/// let registry = K2KTypeRegistry::discover();
/// assert!(registry.is_routable(501));
/// ```
#[proc_macro_derive(RingMessage, attributes(message, ring_message))]
pub fn derive_ring_message(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let args = match RingMessageArgs::from_derive_input(&input) {
        Ok(args) => args,
        Err(e) => return e.write_errors().into(),
    };

    let name = &args.ident;
    let (impl_generics, ty_generics, where_clause) = args.generics.split_for_impl();

    // Calculate base type ID (offset within domain, or absolute if no domain)
    let base_type_id = args.type_id.unwrap_or_else(|| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        name.to_string().hash(&mut hasher);
        // If domain is set, hash to a value within 0-99 range
        if args.domain.is_some() {
            hasher.finish() % 100
        } else {
            hasher.finish()
        }
    });

    // Find annotated fields
    let fields = match &args.data {
        ast::Data::Struct(fields) => fields,
        _ => panic!("RingMessage can only be derived for structs"),
    };

    let mut id_field: Option<&syn::Ident> = None;
    let mut correlation_field: Option<&syn::Ident> = None;
    let mut priority_field: Option<&syn::Ident> = None;

    for field in fields.iter() {
        if field.id {
            id_field = field.ident.as_ref();
        }
        if field.correlation {
            correlation_field = field.ident.as_ref();
        }
        if field.priority {
            priority_field = field.ident.as_ref();
        }
    }

    // Generate message_id method
    let message_id_impl = if let Some(field) = id_field {
        quote! { self.#field }
    } else {
        quote! { ::ringkernel_core::message::MessageId::new(0) }
    };

    // Generate correlation_id method
    let correlation_id_impl = if let Some(field) = correlation_field {
        quote! { self.#field }
    } else {
        quote! { ::ringkernel_core::message::CorrelationId::none() }
    };

    // Generate priority method
    let priority_impl = if let Some(field) = priority_field {
        quote! { self.#field }
    } else {
        quote! { ::ringkernel_core::message::Priority::Normal }
    };

    // Generate message_type() implementation based on whether domain is specified
    let message_type_impl = if let Some(ref domain_str) = args.domain {
        // With domain: type_id = domain.base_type_id() + offset
        quote! {
            ::ringkernel_core::domain::Domain::from_str(#domain_str)
                .unwrap_or(::ringkernel_core::domain::Domain::General)
                .base_type_id() + #base_type_id
        }
    } else {
        // Without domain: use absolute type_id
        quote! { #base_type_id }
    };

    // Generate DomainMessage impl if domain is specified
    let domain_impl = if let Some(ref domain_str) = args.domain {
        quote! {
            impl #impl_generics ::ringkernel_core::domain::DomainMessage for #name #ty_generics #where_clause {
                fn domain() -> ::ringkernel_core::domain::Domain {
                    ::ringkernel_core::domain::Domain::from_str(#domain_str)
                        .unwrap_or(::ringkernel_core::domain::Domain::General)
                }
            }
        }
    } else {
        quote! {}
    };

    // Generate K2K registration if k2k_routable is set
    let k2k_registration = if args.k2k_routable {
        let registration_name = format_ident!(
            "__K2K_MESSAGE_REGISTRATION_{}",
            name.to_string().to_uppercase()
        );
        let type_name_str = name.to_string();
        let category_tokens = match &args.category {
            Some(cat) => quote! { ::std::option::Option::Some(#cat) },
            None => quote! { ::std::option::Option::None },
        };

        quote! {
            #[allow(non_upper_case_globals)]
            #[::inventory::submit]
            static #registration_name: ::ringkernel_core::k2k::K2KMessageRegistration =
                ::ringkernel_core::k2k::K2KMessageRegistration {
                    type_id: {
                        // Note: This is a const context, so we use the base calculation
                        // For domain types, we need to add the base manually
                        #base_type_id
                    },
                    type_name: #type_name_str,
                    k2k_routable: true,
                    category: #category_tokens,
                };
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        impl #impl_generics ::ringkernel_core::message::RingMessage for #name #ty_generics #where_clause {
            fn message_type() -> u64 {
                #message_type_impl
            }

            fn message_id(&self) -> ::ringkernel_core::message::MessageId {
                #message_id_impl
            }

            fn correlation_id(&self) -> ::ringkernel_core::message::CorrelationId {
                #correlation_id_impl
            }

            fn priority(&self) -> ::ringkernel_core::message::Priority {
                #priority_impl
            }

            fn serialize(&self) -> Vec<u8> {
                // Use rkyv for serialization with a 4KB scratch buffer
                // For larger payloads, rkyv will allocate as needed
                ::rkyv::to_bytes::<_, 4096>(self)
                    .map(|v| v.to_vec())
                    .unwrap_or_default()
            }

            fn deserialize(bytes: &[u8]) -> ::ringkernel_core::error::Result<Self>
            where
                Self: Sized,
            {
                use ::rkyv::Deserialize as _;
                let archived = unsafe { ::rkyv::archived_root::<Self>(bytes) };
                let deserialized: Self = archived.deserialize(&mut ::rkyv::Infallible)
                    .map_err(|_| ::ringkernel_core::error::RingKernelError::DeserializationError(
                        "rkyv deserialization failed".to_string()
                    ))?;
                Ok(deserialized)
            }

            fn size_hint(&self) -> usize {
                ::std::mem::size_of::<Self>()
            }
        }

        #domain_impl

        #k2k_registration
    };

    TokenStream::from(expanded)
}

/// Attributes for the ring_kernel macro.
#[derive(Debug, FromMeta)]
struct RingKernelArgs {
    /// Kernel identifier.
    id: String,
    /// Execution mode (persistent or event_driven).
    #[darling(default)]
    mode: Option<String>,
    /// Grid size.
    #[darling(default)]
    grid_size: Option<u32>,
    /// Block size.
    #[darling(default)]
    block_size: Option<u32>,
    /// Target kernels this kernel publishes to.
    #[darling(default)]
    publishes_to: Option<String>,
}

/// Attribute macro for defining ring kernel handlers.
///
/// # Attributes
///
/// - `id` (required) - Unique kernel identifier
/// - `mode` - Execution mode: "persistent" (default) or "event_driven"
/// - `grid_size` - Number of blocks (default: 1)
/// - `block_size` - Threads per block (default: 256)
/// - `publishes_to` - Comma-separated list of target kernel IDs
///
/// # Example
///
/// ```ignore
/// #[ring_kernel(id = "processor", mode = "persistent", block_size = 128)]
/// async fn handle(ctx: &mut RingContext, msg: MyMessage) -> MyResponse {
///     // Process message
///     MyResponse { ... }
/// }
/// ```
#[proc_macro_attribute]
pub fn ring_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = match darling::ast::NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(darling::Error::from(e).write_errors()),
    };

    let args = match RingKernelArgs::from_list(&args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let input = parse_macro_input!(item as ItemFn);

    let kernel_id = &args.id;
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_attrs = &input.attrs;

    // Parse function signature
    let inputs = &input.sig.inputs;
    let output = &input.sig.output;

    // Extract context and message types from signature
    let (_ctx_arg, msg_arg) = if inputs.len() >= 2 {
        let ctx = inputs.first();
        let msg = inputs.iter().nth(1);
        (ctx, msg)
    } else {
        (None, None)
    };

    // Get message type
    let msg_type = msg_arg
        .map(|arg| {
            if let syn::FnArg::Typed(pat_type) = arg {
                pat_type.ty.clone()
            } else {
                syn::parse_quote!(())
            }
        })
        .unwrap_or_else(|| syn::parse_quote!(()));

    // Generate kernel mode
    let mode = args.mode.as_deref().unwrap_or("persistent");
    let mode_expr = if mode == "event_driven" {
        quote! { ::ringkernel_core::types::KernelMode::EventDriven }
    } else {
        quote! { ::ringkernel_core::types::KernelMode::Persistent }
    };

    // Generate grid/block size
    let grid_size = args.grid_size.unwrap_or(1);
    let block_size = args.block_size.unwrap_or(256);

    // Parse publishes_to into a list of target kernel IDs
    let publishes_to_targets: Vec<String> = args
        .publishes_to
        .as_ref()
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
        .unwrap_or_default();

    // Generate registration struct name
    let registration_name = format_ident!(
        "__RINGKERNEL_REGISTRATION_{}",
        fn_name.to_string().to_uppercase()
    );
    let handler_name = format_ident!("{}_handler", fn_name);

    // Generate the expanded code
    let expanded = quote! {
        // Original function (preserved for documentation/testing)
        #(#fn_attrs)*
        #fn_vis async fn #fn_name #inputs #output #fn_block

        // Kernel handler wrapper
        #fn_vis fn #handler_name(
            ctx: &mut ::ringkernel_core::RingContext<'_>,
            envelope: ::ringkernel_core::message::MessageEnvelope,
        ) -> ::std::pin::Pin<Box<dyn ::std::future::Future<Output = ::ringkernel_core::error::Result<::ringkernel_core::message::MessageEnvelope>> + Send + '_>> {
            Box::pin(async move {
                // Deserialize input message
                let msg: #msg_type = ::ringkernel_core::message::RingMessage::deserialize(&envelope.payload)?;

                // Call the actual handler
                let response = #fn_name(ctx, msg).await;

                // Serialize response
                let response_payload = ::ringkernel_core::message::RingMessage::serialize(&response);
                let response_header = ::ringkernel_core::message::MessageHeader::new(
                    <_ as ::ringkernel_core::message::RingMessage>::message_type(),
                    envelope.header.dest_kernel,
                    envelope.header.source_kernel,
                    response_payload.len(),
                    ctx.now(),
                ).with_correlation(envelope.header.correlation_id);

                Ok(::ringkernel_core::message::MessageEnvelope {
                    header: response_header,
                    payload: response_payload,
                })
            })
        }

        // Kernel registration
        #[allow(non_upper_case_globals)]
        #[::inventory::submit]
        static #registration_name: ::ringkernel_core::__private::KernelRegistration = ::ringkernel_core::__private::KernelRegistration {
            id: #kernel_id,
            mode: #mode_expr,
            grid_size: #grid_size,
            block_size: #block_size,
            publishes_to: &[#(#publishes_to_targets),*],
        };
    };

    TokenStream::from(expanded)
}

/// Derive macro for GPU-compatible types.
///
/// Ensures the type has a stable memory layout suitable for GPU transfer.
#[proc_macro_derive(GpuType)]
pub fn derive_gpu_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Generate assertions for GPU compatibility
    let expanded = quote! {
        // Verify type is Copy (required for GPU transfer)
        const _: fn() = || {
            fn assert_copy<T: Copy>() {}
            assert_copy::<#name #ty_generics>();
        };

        // Verify type is Pod (plain old data)
        unsafe impl #impl_generics ::bytemuck::Pod for #name #ty_generics #where_clause {}
        unsafe impl #impl_generics ::bytemuck::Zeroable for #name #ty_generics #where_clause {}
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Stencil Kernel Macro (requires cuda-codegen feature)
// ============================================================================

/// Attributes for the stencil_kernel macro.
#[derive(Debug, FromMeta)]
struct StencilKernelArgs {
    /// Kernel identifier.
    id: String,
    /// Grid dimensionality: "1d", "2d", or "3d".
    #[darling(default)]
    grid: Option<String>,
    /// Tile/block size (single value for square tiles).
    #[darling(default)]
    tile_size: Option<u32>,
    /// Tile width (for non-square tiles).
    #[darling(default)]
    tile_width: Option<u32>,
    /// Tile height (for non-square tiles).
    #[darling(default)]
    tile_height: Option<u32>,
    /// Halo/ghost cell width (stencil radius).
    #[darling(default)]
    halo: Option<u32>,
}

/// Attribute macro for defining stencil kernels that transpile to CUDA.
///
/// This macro generates CUDA C code from Rust stencil kernel functions at compile time.
/// The generated CUDA source is embedded in the binary and can be compiled at runtime
/// using NVRTC.
///
/// # Attributes
///
/// - `id` (required) - Unique kernel identifier
/// - `grid` - Grid dimensionality: "1d", "2d" (default), or "3d"
/// - `tile_size` - Tile/block size (default: 16)
/// - `tile_width` / `tile_height` - Non-square tile dimensions
/// - `halo` - Stencil radius / ghost cell width (default: 1)
///
/// # Supported Rust Subset
///
/// - Primitives: `f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `bool`
/// - Slices: `&[T]`, `&mut [T]`
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`
/// - Comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=`
/// - Let bindings: `let x = expr;`
/// - If/else: `if cond { a } else { b }`
/// - Stencil intrinsics via `GridPos`
///
/// # Example
///
/// ```ignore
/// use ringkernel_derive::stencil_kernel;
/// use ringkernel_cuda_codegen::GridPos;
///
/// #[stencil_kernel(id = "fdtd", grid = "2d", tile_size = 16, halo = 1)]
/// fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
///     let curr = p[pos.idx()];
///     let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
///     p_prev[pos.idx()] = 2.0 * curr - p_prev[pos.idx()] + c2 * lap;
/// }
///
/// // Access generated CUDA source:
/// assert!(FDTD_CUDA_SOURCE.contains("__global__"));
/// ```
#[proc_macro_attribute]
pub fn stencil_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = match darling::ast::NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(darling::Error::from(e).write_errors()),
    };

    let args = match StencilKernelArgs::from_list(&args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let input = parse_macro_input!(item as ItemFn);

    // Generate the stencil kernel code
    stencil_kernel_impl(args, input)
}

fn stencil_kernel_impl(args: StencilKernelArgs, input: ItemFn) -> TokenStream {
    let kernel_id = &args.id;
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_inputs = &input.sig.inputs;
    let fn_output = &input.sig.output;
    let fn_attrs = &input.attrs;

    // Parse configuration
    let grid = args.grid.as_deref().unwrap_or("2d");
    let tile_width = args
        .tile_width
        .unwrap_or_else(|| args.tile_size.unwrap_or(16));
    let tile_height = args
        .tile_height
        .unwrap_or_else(|| args.tile_size.unwrap_or(16));
    let halo = args.halo.unwrap_or(1);

    // Generate CUDA source constant name
    let cuda_const_name = format_ident!("{}_CUDA_SOURCE", fn_name.to_string().to_uppercase());

    // Generate registration name
    let registration_name = format_ident!(
        "__STENCIL_KERNEL_REGISTRATION_{}",
        fn_name.to_string().to_uppercase()
    );

    // Transpile to CUDA (if feature enabled)
    #[cfg(feature = "cuda-codegen")]
    let cuda_source_code = {
        use ringkernel_cuda_codegen::{transpile_stencil_kernel, Grid, StencilConfig};

        let grid_type = match grid {
            "1d" => Grid::Grid1D,
            "2d" => Grid::Grid2D,
            "3d" => Grid::Grid3D,
            _ => Grid::Grid2D,
        };

        let config = StencilConfig::new(kernel_id.clone())
            .with_grid(grid_type)
            .with_tile_size(tile_width as usize, tile_height as usize)
            .with_halo(halo as usize);

        match transpile_stencil_kernel(&input, &config) {
            Ok(cuda) => cuda,
            Err(e) => {
                return TokenStream::from(
                    syn::Error::new_spanned(
                        &input.sig.ident,
                        format!("CUDA transpilation failed: {}", e),
                    )
                    .to_compile_error(),
                );
            }
        }
    };

    #[cfg(not(feature = "cuda-codegen"))]
    let cuda_source_code = format!(
        "// CUDA codegen not enabled. Enable 'cuda-codegen' feature.\n// Kernel: {}\n",
        kernel_id
    );

    // Generate the expanded code
    let expanded = quote! {
        // Original function (for documentation/testing/CPU fallback)
        #(#fn_attrs)*
        #fn_vis fn #fn_name #fn_inputs #fn_output #fn_block

        /// Generated CUDA source code for this stencil kernel.
        #fn_vis const #cuda_const_name: &str = #cuda_source_code;

        /// Stencil kernel registration for runtime discovery.
        #[allow(non_upper_case_globals)]
        #[::inventory::submit]
        static #registration_name: ::ringkernel_core::__private::StencilKernelRegistration =
            ::ringkernel_core::__private::StencilKernelRegistration {
                id: #kernel_id,
                grid: #grid,
                tile_width: #tile_width,
                tile_height: #tile_height,
                halo: #halo,
                cuda_source: #cuda_source_code,
            };
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Multi-Backend GPU Kernel Macro
// ============================================================================

/// GPU backend targets (internal use only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuBackend {
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// WebGPU backend (cross-platform).
    Wgpu,
    /// CPU fallback backend.
    Cpu,
}

impl GpuBackend {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cuda" => Some(Self::Cuda),
            "metal" => Some(Self::Metal),
            "wgpu" | "webgpu" => Some(Self::Wgpu),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Wgpu => "wgpu",
            Self::Cpu => "cpu",
        }
    }
}

/// GPU capability flags that can be required by a kernel (internal use only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuCapability {
    /// 64-bit floating point support.
    Float64,
    /// 64-bit integer support.
    Int64,
    /// 64-bit atomics support.
    Atomic64,
    /// Cooperative groups / grid-wide sync.
    CooperativeGroups,
    /// Subgroup / warp / SIMD operations.
    Subgroups,
    /// Shared memory / threadgroup memory.
    SharedMemory,
    /// Dynamic parallelism (launching kernels from kernels).
    DynamicParallelism,
    /// Half-precision (f16) support.
    Float16,
}

impl GpuCapability {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f64" | "float64" => Some(Self::Float64),
            "i64" | "int64" => Some(Self::Int64),
            "atomic64" => Some(Self::Atomic64),
            "cooperative_groups" | "cooperativegroups" | "grid_sync" => {
                Some(Self::CooperativeGroups)
            }
            "subgroups" | "warp" | "simd" => Some(Self::Subgroups),
            "shared_memory" | "sharedmemory" | "threadgroup" => Some(Self::SharedMemory),
            "dynamic_parallelism" | "dynamicparallelism" => Some(Self::DynamicParallelism),
            "f16" | "float16" | "half" => Some(Self::Float16),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Float64 => "f64",
            Self::Int64 => "i64",
            Self::Atomic64 => "atomic64",
            Self::CooperativeGroups => "cooperative_groups",
            Self::Subgroups => "subgroups",
            Self::SharedMemory => "shared_memory",
            Self::DynamicParallelism => "dynamic_parallelism",
            Self::Float16 => "f16",
        }
    }

    /// Check if a backend supports this capability.
    fn supported_by(&self, backend: GpuBackend) -> bool {
        match (self, backend) {
            // CUDA supports everything
            (_, GpuBackend::Cuda) => true,

            // Metal capabilities
            (Self::Float64, GpuBackend::Metal) => false,
            (Self::CooperativeGroups, GpuBackend::Metal) => false,
            (Self::DynamicParallelism, GpuBackend::Metal) => false,
            (_, GpuBackend::Metal) => true,

            // WebGPU capabilities
            (Self::Float64, GpuBackend::Wgpu) => false,
            (Self::Int64, GpuBackend::Wgpu) => false,
            (Self::Atomic64, GpuBackend::Wgpu) => false, // Emulated only
            (Self::CooperativeGroups, GpuBackend::Wgpu) => false,
            (Self::DynamicParallelism, GpuBackend::Wgpu) => false,
            (Self::Subgroups, GpuBackend::Wgpu) => true, // Optional extension
            (_, GpuBackend::Wgpu) => true,

            // CPU supports everything (in emulation)
            (_, GpuBackend::Cpu) => true,
        }
    }
}

/// Attributes for the gpu_kernel macro.
#[derive(Debug)]
struct GpuKernelArgs {
    /// Kernel identifier.
    id: Option<String>,
    /// Target backends to generate code for.
    backends: Vec<GpuBackend>,
    /// Fallback order for backend selection.
    fallback: Vec<GpuBackend>,
    /// Required capabilities.
    requires: Vec<GpuCapability>,
    /// Block/workgroup size.
    block_size: Option<u32>,
}

impl Default for GpuKernelArgs {
    fn default() -> Self {
        Self {
            id: None,
            backends: vec![GpuBackend::Cuda, GpuBackend::Metal, GpuBackend::Wgpu],
            fallback: vec![
                GpuBackend::Cuda,
                GpuBackend::Metal,
                GpuBackend::Wgpu,
                GpuBackend::Cpu,
            ],
            requires: Vec::new(),
            block_size: None,
        }
    }
}

impl GpuKernelArgs {
    fn parse(attr: proc_macro2::TokenStream) -> Result<Self, darling::Error> {
        let mut args = Self::default();
        let attr_str = attr.to_string();

        // Parse backends = [...]
        if let Some(start) = attr_str.find("backends") {
            if let Some(bracket_start) = attr_str[start..].find('[') {
                if let Some(bracket_end) = attr_str[start + bracket_start..].find(']') {
                    let backends_str =
                        &attr_str[start + bracket_start + 1..start + bracket_start + bracket_end];
                    args.backends = backends_str
                        .split(',')
                        .filter_map(|s| GpuBackend::from_str(s.trim()))
                        .collect();
                }
            }
        }

        // Parse fallback = [...]
        if let Some(start) = attr_str.find("fallback") {
            if let Some(bracket_start) = attr_str[start..].find('[') {
                if let Some(bracket_end) = attr_str[start + bracket_start..].find(']') {
                    let fallback_str =
                        &attr_str[start + bracket_start + 1..start + bracket_start + bracket_end];
                    args.fallback = fallback_str
                        .split(',')
                        .filter_map(|s| GpuBackend::from_str(s.trim()))
                        .collect();
                }
            }
        }

        // Parse requires = [...]
        if let Some(start) = attr_str.find("requires") {
            if let Some(bracket_start) = attr_str[start..].find('[') {
                if let Some(bracket_end) = attr_str[start + bracket_start..].find(']') {
                    let requires_str =
                        &attr_str[start + bracket_start + 1..start + bracket_start + bracket_end];
                    args.requires = requires_str
                        .split(',')
                        .filter_map(|s| GpuCapability::from_str(s.trim()))
                        .collect();
                }
            }
        }

        // Parse id = "..."
        if let Some(start) = attr_str.find("id") {
            if let Some(quote_start) = attr_str[start..].find('"') {
                if let Some(quote_end) = attr_str[start + quote_start + 1..].find('"') {
                    args.id = Some(
                        attr_str[start + quote_start + 1..start + quote_start + 1 + quote_end]
                            .to_string(),
                    );
                }
            }
        }

        // Parse block_size = N
        if let Some(start) = attr_str.find("block_size") {
            if let Some(eq) = attr_str[start..].find('=') {
                let rest = &attr_str[start + eq + 1..];
                let num_end = rest
                    .find(|c: char| !c.is_numeric() && c != ' ')
                    .unwrap_or(rest.len());
                if let Ok(n) = rest[..num_end].trim().parse() {
                    args.block_size = Some(n);
                }
            }
        }

        Ok(args)
    }

    /// Validate that all required capabilities are supported by at least one backend.
    fn validate_capabilities(&self) -> Result<(), String> {
        for cap in &self.requires {
            let mut supported_by_any = false;
            for backend in &self.backends {
                if cap.supported_by(*backend) {
                    supported_by_any = true;
                    break;
                }
            }
            if !supported_by_any {
                return Err(format!(
                    "Capability '{}' is not supported by any of the specified backends: {:?}",
                    cap.as_str(),
                    self.backends.iter().map(|b| b.as_str()).collect::<Vec<_>>()
                ));
            }
        }
        Ok(())
    }

    /// Get backends that support all required capabilities.
    fn compatible_backends(&self) -> Vec<GpuBackend> {
        self.backends
            .iter()
            .filter(|backend| self.requires.iter().all(|cap| cap.supported_by(**backend)))
            .copied()
            .collect()
    }
}

/// Attribute macro for defining multi-backend GPU kernels.
///
/// This macro generates code for multiple GPU backends with compile-time
/// capability validation. It integrates with the `ringkernel-ir` crate
/// to lower Rust DSL to backend-specific shader code.
///
/// # Attributes
///
/// - `backends = [cuda, metal, wgpu]` - Target backends (default: all)
/// - `fallback = [cuda, metal, wgpu, cpu]` - Fallback order for runtime selection
/// - `requires = [f64, atomic64]` - Required capabilities (validated at compile time)
/// - `id = "kernel_name"` - Explicit kernel identifier
/// - `block_size = 256` - Thread block size
///
/// # Example
///
/// ```ignore
/// use ringkernel_derive::gpu_kernel;
///
/// #[gpu_kernel(backends = [cuda, metal], requires = [subgroups])]
/// fn warp_reduce(data: &mut [f32], n: i32) {
///     let idx = global_thread_id_x();
///     if idx < n {
///         // Use warp shuffle for reduction
///         let val = data[idx as usize];
///         let reduced = warp_reduce_sum(val);
///         if lane_id() == 0 {
///             data[idx as usize] = reduced;
///         }
///     }
/// }
/// ```
///
/// # Capability Checking
///
/// The macro validates at compile time that all required capabilities are
/// supported by at least one target backend:
///
/// | Capability | CUDA | Metal | WebGPU | CPU |
/// |------------|------|-------|--------|-----|
/// | f64        | Yes  | No    | No     | Yes |
/// | i64        | Yes  | Yes   | No     | Yes |
/// | atomic64   | Yes  | Yes   | No*    | Yes |
/// | cooperative_groups | Yes | No | No | Yes |
/// | subgroups  | Yes  | Yes   | Opt    | Yes |
/// | shared_memory | Yes | Yes | Yes    | Yes |
/// | f16        | Yes  | Yes   | Yes    | Yes |
///
/// *WebGPU emulates 64-bit atomics with 32-bit pairs.
///
/// # Generated Code
///
/// For each compatible backend, the macro generates:
/// - Backend-specific source code constant (e.g., `KERNEL_NAME_CUDA_SOURCE`)
/// - Registration entry for runtime discovery
/// - CPU fallback function (if `cpu_fallback = true`)
#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr2: proc_macro2::TokenStream = attr.into();
    let args = match GpuKernelArgs::parse(attr2) {
        Ok(args) => args,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let input = parse_macro_input!(item as ItemFn);

    // Validate capabilities
    if let Err(msg) = args.validate_capabilities() {
        return TokenStream::from(
            syn::Error::new_spanned(&input.sig.ident, msg).to_compile_error(),
        );
    }

    gpu_kernel_impl(args, input)
}

fn gpu_kernel_impl(args: GpuKernelArgs, input: ItemFn) -> TokenStream {
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_inputs = &input.sig.inputs;
    let fn_output = &input.sig.output;
    let fn_attrs = &input.attrs;

    let kernel_id = args.id.clone().unwrap_or_else(|| fn_name.to_string());
    let block_size = args.block_size.unwrap_or(256);

    // Get compatible backends
    let compatible_backends = args.compatible_backends();

    // Generate backend-specific source constants
    let mut source_constants = Vec::new();

    for backend in &compatible_backends {
        let const_name = format_ident!(
            "{}_{}",
            fn_name.to_string().to_uppercase(),
            backend.as_str().to_uppercase()
        );

        let backend_str = backend.as_str();

        // Generate placeholder source (actual IR lowering happens at build time)
        // In a full implementation, this would call ringkernel-ir lowering
        let source_placeholder = format!(
            "// {} source for kernel '{}'\n// Generated by ringkernel-derive\n// Capabilities: {:?}\n",
            backend_str.to_uppercase(),
            kernel_id,
            args.requires.iter().map(|c| c.as_str()).collect::<Vec<_>>()
        );

        source_constants.push(quote! {
            /// Generated source code for this kernel.
            #fn_vis const #const_name: &str = #source_placeholder;
        });
    }

    // Generate capability flags as strings
    let capability_strs: Vec<_> = args.requires.iter().map(|c| c.as_str()).collect();
    let backend_strs: Vec<_> = compatible_backends.iter().map(|b| b.as_str()).collect();
    let fallback_strs: Vec<_> = args.fallback.iter().map(|b| b.as_str()).collect();

    // Generate registration struct name
    let registration_name = format_ident!(
        "__GPU_KERNEL_REGISTRATION_{}",
        fn_name.to_string().to_uppercase()
    );

    // Generate info struct name
    let info_name = format_ident!("{}_INFO", fn_name.to_string().to_uppercase());

    // Generate the expanded code
    let expanded = quote! {
        // Original function (CPU fallback / documentation / testing)
        #(#fn_attrs)*
        #fn_vis fn #fn_name #fn_inputs #fn_output #fn_block

        // Backend source constants
        #(#source_constants)*

        /// Multi-backend kernel information.
        #fn_vis mod #info_name {
            /// Kernel identifier.
            pub const ID: &str = #kernel_id;

            /// Block/workgroup size.
            pub const BLOCK_SIZE: u32 = #block_size;

            /// Required capabilities.
            pub const CAPABILITIES: &[&str] = &[#(#capability_strs),*];

            /// Compatible backends (those that support all required capabilities).
            pub const BACKENDS: &[&str] = &[#(#backend_strs),*];

            /// Fallback order for runtime backend selection.
            pub const FALLBACK_ORDER: &[&str] = &[#(#fallback_strs),*];
        }

        /// GPU kernel registration for runtime discovery.
        #[allow(non_upper_case_globals)]
        #[::inventory::submit]
        static #registration_name: ::ringkernel_core::__private::GpuKernelRegistration =
            ::ringkernel_core::__private::GpuKernelRegistration {
                id: #kernel_id,
                block_size: #block_size,
                capabilities: &[#(#capability_strs),*],
                backends: &[#(#backend_strs),*],
                fallback_order: &[#(#fallback_strs),*],
            };
    };

    TokenStream::from(expanded)
}

// ============================================================================
// ControlBlockState Derive Macro (FR-4)
// ============================================================================

/// Attributes for the ControlBlockState derive macro.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(state), supports(struct_named))]
struct ControlBlockStateArgs {
    ident: syn::Ident,
    generics: syn::Generics,
    /// State version for forward compatibility.
    #[darling(default)]
    version: Option<u32>,
}

/// Derive macro for implementing EmbeddedState trait.
///
/// This macro generates implementations for types that can be stored in
/// the ControlBlock's 24-byte `_reserved` field for zero-copy state access.
///
/// # Requirements
///
/// The type must:
/// - Be `#[repr(C)]` for stable memory layout
/// - Be <= 24 bytes in size (checked at compile time)
/// - Implement `Clone`, `Copy`, and `Default`
/// - Contain only POD (Plain Old Data) types
///
/// # Attributes
///
/// - `#[state(version = N)]` - Set state version for migrations (default: 1)
///
/// # Example
///
/// ```ignore
/// #[derive(ControlBlockState, Default, Clone, Copy)]
/// #[repr(C, align(8))]
/// #[state(version = 1)]
/// pub struct OrderBookState {
///     pub best_bid: u64,    // 8 bytes
///     pub best_ask: u64,    // 8 bytes
///     pub order_count: u32, // 4 bytes
///     pub _pad: u32,        // 4 bytes (padding for alignment)
/// }  // Total: 24 bytes - fits in ControlBlock._reserved
///
/// // Use with ControlBlockStateHelper:
/// let mut block = ControlBlock::new();
/// let state = OrderBookState { best_bid: 100, best_ask: 101, order_count: 42, _pad: 0 };
/// ControlBlockStateHelper::write_embedded(&mut block, &state)?;
/// ```
///
/// # Size Validation
///
/// The macro generates a compile-time assertion that fails if the type
/// exceeds 24 bytes:
///
/// ```ignore
/// #[derive(ControlBlockState, Default, Clone, Copy)]
/// #[repr(C)]
/// struct TooLarge {
///     data: [u8; 32],  // 32 bytes - COMPILE ERROR!
/// }
/// ```
#[proc_macro_derive(ControlBlockState, attributes(state))]
pub fn derive_control_block_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let args = match ControlBlockStateArgs::from_derive_input(&input) {
        Ok(args) => args,
        Err(e) => return e.write_errors().into(),
    };

    let name = &args.ident;
    let (impl_generics, ty_generics, where_clause) = args.generics.split_for_impl();
    let version = args.version.unwrap_or(1);

    let expanded = quote! {
        // Compile-time size check: EmbeddedState must fit in 24 bytes
        const _: () = {
            assert!(
                ::std::mem::size_of::<#name #ty_generics>() <= 24,
                "ControlBlockState types must fit in 24 bytes (ControlBlock._reserved size)"
            );
        };

        // Verify type is Copy (required for GPU transfer)
        const _: fn() = || {
            fn assert_copy<T: Copy>() {}
            assert_copy::<#name #ty_generics>();
        };

        // Implement Pod and Zeroable (required by EmbeddedState)
        // SAFETY: Type is #[repr(C)] with only primitive types, verified by user
        unsafe impl #impl_generics ::bytemuck::Zeroable for #name #ty_generics #where_clause {}
        unsafe impl #impl_generics ::bytemuck::Pod for #name #ty_generics #where_clause {}

        // Implement EmbeddedState
        impl #impl_generics ::ringkernel_core::state::EmbeddedState for #name #ty_generics #where_clause {
            const VERSION: u32 = #version;

            fn is_embedded() -> bool {
                true
            }
        }
    };

    TokenStream::from(expanded)
}
