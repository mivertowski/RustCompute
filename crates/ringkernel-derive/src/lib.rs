//! Procedural macros for RingKernel.
//!
//! This crate provides two main macros:
//!
//! - `#[derive(RingMessage)]` - Implement the RingMessage trait for message types
//! - `#[ring_kernel]` - Define a ring kernel handler
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

use darling::{ast, FromDeriveInput, FromField, FromMeta};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, DeriveInput, ItemFn};

/// Attributes for the RingMessage derive macro.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(message), supports(struct_named))]
struct RingMessageArgs {
    ident: syn::Ident,
    generics: syn::Generics,
    data: ast::Data<(), RingMessageField>,
    /// Optional explicit message type ID.
    #[darling(default)]
    type_id: Option<u64>,
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
/// On the struct:
/// - `#[message(type_id = 123)]` - Set explicit message type ID
///
/// On fields:
/// - `#[message(id)]` - Mark as message ID field
/// - `#[message(correlation)]` - Mark as correlation ID field
/// - `#[message(priority)]` - Mark as priority field
///
/// # Example
///
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
#[proc_macro_derive(RingMessage, attributes(message))]
pub fn derive_ring_message(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let args = match RingMessageArgs::from_derive_input(&input) {
        Ok(args) => args,
        Err(e) => return e.write_errors().into(),
    };

    let name = &args.ident;
    let (impl_generics, ty_generics, where_clause) = args.generics.split_for_impl();

    // Calculate type ID from name hash if not specified
    let type_id = args.type_id.unwrap_or_else(|| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        name.to_string().hash(&mut hasher);
        hasher.finish()
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

    let expanded = quote! {
        impl #impl_generics ::ringkernel_core::message::RingMessage for #name #ty_generics #where_clause {
            fn message_type() -> u64 {
                #type_id
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
