//! Type mapping from Rust to WGSL.
//!
//! This module handles the conversion of Rust types to their WGSL equivalents.
//!
//! # Type Mappings
//!
//! | Rust Type | WGSL Type | Notes |
//! |-----------|-----------|-------|
//! | `f32` | `f32` | Direct mapping |
//! | `f64` | `f32` | **Warning**: Downcast, WGSL 1.0 has no f64 |
//! | `i32` | `i32` | Direct mapping |
//! | `u32` | `u32` | Direct mapping |
//! | `i64` | `vec2<i32>` | Emulated as lo/hi pair |
//! | `u64` | `vec2<u32>` | Emulated as lo/hi pair |
//! | `bool` | `bool` | Direct mapping |
//! | `&[T]` | `array<T>` | Storage buffer binding |
//! | `&mut [T]` | `array<T>` | Storage buffer binding (read_write) |

use std::collections::HashMap;

/// WGSL address spaces for variable declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSpace {
    /// `var<function>` - Local function scope
    Function,
    /// `var<private>` - Thread-private global
    Private,
    /// `var<workgroup>` - Shared within workgroup
    Workgroup,
    /// `var<uniform>` - Uniform buffer
    Uniform,
    /// `var<storage>` - Storage buffer
    Storage,
}

impl AddressSpace {
    /// Convert to WGSL syntax.
    pub fn to_wgsl(&self) -> &'static str {
        match self {
            AddressSpace::Function => "function",
            AddressSpace::Private => "private",
            AddressSpace::Workgroup => "workgroup",
            AddressSpace::Uniform => "uniform",
            AddressSpace::Storage => "storage",
        }
    }
}

/// Access mode for storage/uniform buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessMode {
    /// Read-only access
    #[default]
    Read,
    /// Write-only access (rare in WGSL)
    Write,
    /// Read and write access
    ReadWrite,
}

impl AccessMode {
    /// Convert to WGSL syntax.
    pub fn to_wgsl(&self) -> &'static str {
        match self {
            AccessMode::Read => "read",
            AccessMode::Write => "write",
            AccessMode::ReadWrite => "read_write",
        }
    }
}

/// WGSL type representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgslType {
    /// 32-bit float
    F32,
    /// 32-bit signed integer
    I32,
    /// 32-bit unsigned integer
    U32,
    /// Boolean
    Bool,
    /// Void (for functions with no return)
    Void,
    /// 2-component vector
    Vec2(Box<WgslType>),
    /// 3-component vector
    Vec3(Box<WgslType>),
    /// 4-component vector
    Vec4(Box<WgslType>),
    /// 2x2 matrix
    Mat2x2(Box<WgslType>),
    /// 3x3 matrix
    Mat3x3(Box<WgslType>),
    /// 4x4 matrix
    Mat4x4(Box<WgslType>),
    /// Array type
    Array {
        element: Box<WgslType>,
        /// None means runtime-sized array
        size: Option<usize>,
    },
    /// Pointer type (for function parameters)
    Ptr {
        address_space: AddressSpace,
        inner: Box<WgslType>,
        access: AccessMode,
    },
    /// Atomic type
    Atomic(Box<WgslType>),
    /// User-defined struct
    Struct(String),
    /// Emulated 64-bit unsigned (stored as vec2<u32>)
    U64Pair,
    /// Emulated 64-bit signed (stored as vec2<i32>)
    I64Pair,
}

impl WgslType {
    /// Convert to WGSL type syntax.
    pub fn to_wgsl(&self) -> String {
        match self {
            WgslType::F32 => "f32".to_string(),
            WgslType::I32 => "i32".to_string(),
            WgslType::U32 => "u32".to_string(),
            WgslType::Bool => "bool".to_string(),
            WgslType::Void => "".to_string(), // Functions with no return type
            WgslType::Vec2(inner) => format!("vec2<{}>", inner.to_wgsl()),
            WgslType::Vec3(inner) => format!("vec3<{}>", inner.to_wgsl()),
            WgslType::Vec4(inner) => format!("vec4<{}>", inner.to_wgsl()),
            WgslType::Mat2x2(inner) => format!("mat2x2<{}>", inner.to_wgsl()),
            WgslType::Mat3x3(inner) => format!("mat3x3<{}>", inner.to_wgsl()),
            WgslType::Mat4x4(inner) => format!("mat4x4<{}>", inner.to_wgsl()),
            WgslType::Array { element, size } => {
                match size {
                    Some(n) => format!("array<{}, {}>", element.to_wgsl(), n),
                    None => format!("array<{}>", element.to_wgsl()),
                }
            }
            WgslType::Ptr { address_space, inner, access } => {
                format!(
                    "ptr<{}, {}, {}>",
                    address_space.to_wgsl(),
                    inner.to_wgsl(),
                    access.to_wgsl()
                )
            }
            WgslType::Atomic(inner) => format!("atomic<{}>", inner.to_wgsl()),
            WgslType::Struct(name) => name.clone(),
            WgslType::U64Pair => "vec2<u32>".to_string(),
            WgslType::I64Pair => "vec2<i32>".to_string(),
        }
    }

    /// Check if this type is a 64-bit emulated type.
    pub fn is_emulated_64bit(&self) -> bool {
        matches!(self, WgslType::U64Pair | WgslType::I64Pair)
    }

    /// Check if this type is a scalar.
    pub fn is_scalar(&self) -> bool {
        matches!(self, WgslType::F32 | WgslType::I32 | WgslType::U32 | WgslType::Bool)
    }

    /// Check if this type is a vector.
    pub fn is_vector(&self) -> bool {
        matches!(self, WgslType::Vec2(_) | WgslType::Vec3(_) | WgslType::Vec4(_))
    }

    /// Get the element type for arrays and vectors.
    pub fn element_type(&self) -> Option<&WgslType> {
        match self {
            WgslType::Array { element, .. } => Some(element),
            WgslType::Vec2(e) | WgslType::Vec3(e) | WgslType::Vec4(e) => Some(e),
            WgslType::Atomic(inner) => Some(inner),
            _ => None,
        }
    }
}

/// Type mapper for converting Rust types to WGSL types.
#[derive(Debug, Clone)]
pub struct TypeMapper {
    /// Custom type mappings (Rust type name -> WGSL type)
    custom_types: HashMap<String, WgslType>,
    /// Whether to emit warnings for lossy conversions
    warn_on_lossy: bool,
}

impl Default for TypeMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeMapper {
    /// Create a new type mapper with default mappings.
    pub fn new() -> Self {
        Self {
            custom_types: HashMap::new(),
            warn_on_lossy: true,
        }
    }

    /// Register a custom type mapping.
    pub fn register_type(&mut self, rust_name: &str, wgsl_type: WgslType) {
        self.custom_types.insert(rust_name.to_string(), wgsl_type);
    }

    /// Disable warnings for lossy conversions (e.g., f64 -> f32).
    pub fn disable_lossy_warnings(&mut self) {
        self.warn_on_lossy = false;
    }

    /// Map a Rust type to a WGSL type.
    pub fn map_type(&self, ty: &syn::Type) -> Result<WgslType, String> {
        match ty {
            syn::Type::Path(type_path) => self.map_type_path(type_path),
            syn::Type::Reference(type_ref) => self.map_reference(type_ref),
            syn::Type::Array(type_array) => self.map_array(type_array),
            syn::Type::Slice(type_slice) => self.map_slice(type_slice),
            syn::Type::Tuple(tuple) if tuple.elems.is_empty() => Ok(WgslType::Void),
            _ => Err(format!("Unsupported type: {:?}", ty)),
        }
    }

    fn map_type_path(&self, type_path: &syn::TypePath) -> Result<WgslType, String> {
        let path = &type_path.path;

        // Get the last segment (e.g., "f32" from "std::f32")
        let segment = path.segments.last()
            .ok_or_else(|| "Empty type path".to_string())?;

        let ident = segment.ident.to_string();

        // Check custom types first
        if let Some(wgsl_type) = self.custom_types.get(&ident) {
            return Ok(wgsl_type.clone());
        }

        // Built-in type mappings
        match ident.as_str() {
            "f32" => Ok(WgslType::F32),
            "f64" => {
                if self.warn_on_lossy {
                    eprintln!("Warning: f64 will be downcast to f32 (WGSL 1.0 has no f64)");
                }
                Ok(WgslType::F32) // Downcast with warning
            }
            "i32" => Ok(WgslType::I32),
            "u32" => Ok(WgslType::U32),
            "i64" => Ok(WgslType::I64Pair), // Emulated
            "u64" => Ok(WgslType::U64Pair), // Emulated
            "bool" => Ok(WgslType::Bool),
            "usize" => Ok(WgslType::U32), // WGSL uses 32-bit addressing
            "isize" => Ok(WgslType::I32),

            // Vector types (if we support them)
            "Vec2" | "vec2" => {
                let inner = self.extract_generic_arg(segment)?;
                Ok(WgslType::Vec2(Box::new(inner)))
            }
            "Vec3" | "vec3" => {
                let inner = self.extract_generic_arg(segment)?;
                Ok(WgslType::Vec3(Box::new(inner)))
            }
            "Vec4" | "vec4" => {
                let inner = self.extract_generic_arg(segment)?;
                Ok(WgslType::Vec4(Box::new(inner)))
            }

            // Special marker types (removed during transpilation)
            "GridPos" => Err("GridPos is a marker type".to_string()),
            "RingContext" => Err("RingContext is a marker type".to_string()),

            // Assume user-defined struct
            _ => Ok(WgslType::Struct(ident)),
        }
    }

    fn map_reference(&self, type_ref: &syn::TypeReference) -> Result<WgslType, String> {
        let inner = self.map_type(&type_ref.elem)?;
        let is_mutable = type_ref.mutability.is_some();

        // Check if it's a slice reference
        if let syn::Type::Slice(_) = type_ref.elem.as_ref() {
            // Slice references become storage buffer arrays
            let access = if is_mutable {
                AccessMode::ReadWrite
            } else {
                AccessMode::Read
            };

            // For function parameters, we use ptr<storage, ...>
            // For bindings, we use var<storage, ...>
            Ok(WgslType::Ptr {
                address_space: AddressSpace::Storage,
                inner: Box::new(inner),
                access,
            })
        } else {
            // Regular references become pointers
            let access = if is_mutable {
                AccessMode::ReadWrite
            } else {
                AccessMode::Read
            };

            Ok(WgslType::Ptr {
                address_space: AddressSpace::Function,
                inner: Box::new(inner),
                access,
            })
        }
    }

    fn map_array(&self, type_array: &syn::TypeArray) -> Result<WgslType, String> {
        let element = self.map_type(&type_array.elem)?;

        // Extract the array length
        let size = match &type_array.len {
            syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit), .. }) => {
                lit.base10_parse::<usize>().map_err(|e| e.to_string())?
            }
            _ => return Err("Array length must be a literal integer".to_string()),
        };

        Ok(WgslType::Array {
            element: Box::new(element),
            size: Some(size),
        })
    }

    fn map_slice(&self, type_slice: &syn::TypeSlice) -> Result<WgslType, String> {
        let element = self.map_type(&type_slice.elem)?;

        Ok(WgslType::Array {
            element: Box::new(element),
            size: None, // Runtime-sized
        })
    }

    fn extract_generic_arg(&self, segment: &syn::PathSegment) -> Result<WgslType, String> {
        match &segment.arguments {
            syn::PathArguments::AngleBracketed(args) => {
                if let Some(syn::GenericArgument::Type(ty)) = args.args.first() {
                    self.map_type(ty)
                } else {
                    Err("Expected type argument".to_string())
                }
            }
            _ => Err("Expected angle-bracketed arguments".to_string()),
        }
    }
}

/// Check if a type is the GridPos marker type.
pub fn is_grid_pos_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "GridPos";
        }
    }
    false
}

/// Check if a type is the RingContext marker type.
pub fn is_ring_context_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "RingContext";
        }
    }
    false
}

/// Check if a type is a mutable reference.
pub fn is_mutable_reference(ty: &syn::Type) -> bool {
    if let syn::Type::Reference(type_ref) = ty {
        return type_ref.mutability.is_some();
    }
    false
}

/// Get the element type of a slice type.
pub fn get_slice_element_type(ty: &syn::Type, mapper: &TypeMapper) -> Option<WgslType> {
    if let syn::Type::Reference(type_ref) = ty {
        if let syn::Type::Slice(type_slice) = type_ref.elem.as_ref() {
            return mapper.map_type(&type_slice.elem).ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_primitive_types() {
        let mapper = TypeMapper::new();

        let ty: syn::Type = parse_quote!(f32);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::F32);

        let ty: syn::Type = parse_quote!(i32);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::I32);

        let ty: syn::Type = parse_quote!(u32);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::U32);

        let ty: syn::Type = parse_quote!(bool);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::Bool);
    }

    #[test]
    fn test_64bit_emulation() {
        let mapper = TypeMapper::new();

        let ty: syn::Type = parse_quote!(u64);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::U64Pair);

        let ty: syn::Type = parse_quote!(i64);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::I64Pair);
    }

    #[test]
    fn test_slice_types() {
        let mapper = TypeMapper::new();

        let ty: syn::Type = parse_quote!(&[f32]);
        let result = mapper.map_type(&ty).unwrap();
        assert!(matches!(result, WgslType::Ptr { access: AccessMode::Read, .. }));

        let ty: syn::Type = parse_quote!(&mut [f32]);
        let result = mapper.map_type(&ty).unwrap();
        assert!(matches!(result, WgslType::Ptr { access: AccessMode::ReadWrite, .. }));
    }

    #[test]
    fn test_wgsl_output() {
        assert_eq!(WgslType::F32.to_wgsl(), "f32");
        assert_eq!(WgslType::Vec2(Box::new(WgslType::F32)).to_wgsl(), "vec2<f32>");
        assert_eq!(WgslType::U64Pair.to_wgsl(), "vec2<u32>");
        assert_eq!(
            WgslType::Array { element: Box::new(WgslType::F32), size: Some(16) }.to_wgsl(),
            "array<f32, 16>"
        );
        assert_eq!(
            WgslType::Array { element: Box::new(WgslType::F32), size: None }.to_wgsl(),
            "array<f32>"
        );
        assert_eq!(WgslType::Atomic(Box::new(WgslType::U32)).to_wgsl(), "atomic<u32>");
    }

    #[test]
    fn test_custom_types() {
        let mut mapper = TypeMapper::new();
        mapper.register_type("MyStruct", WgslType::Struct("MyStruct".to_string()));

        let ty: syn::Type = parse_quote!(MyStruct);
        assert_eq!(mapper.map_type(&ty).unwrap(), WgslType::Struct("MyStruct".to_string()));
    }

    #[test]
    fn test_grid_pos_detection() {
        let ty: syn::Type = parse_quote!(GridPos);
        assert!(is_grid_pos_type(&ty));

        let ty: syn::Type = parse_quote!(f32);
        assert!(!is_grid_pos_type(&ty));
    }

    #[test]
    fn test_ring_context_detection() {
        let ty: syn::Type = parse_quote!(RingContext);
        assert!(is_ring_context_type(&ty));

        let ty: syn::Type = parse_quote!(&RingContext);
        // Reference to RingContext should also be detected
        if let syn::Type::Reference(r) = &ty {
            assert!(is_ring_context_type(&r.elem));
        }
    }
}
