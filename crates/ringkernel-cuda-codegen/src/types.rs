//! Type mapping from Rust to CUDA.
//!
//! This module handles the translation of Rust types to their CUDA equivalents.

use crate::{Result, TranspileError};
use syn::{Type, TypePath, TypeReference};

/// CUDA type representation.
#[derive(Debug, Clone, PartialEq)]
pub enum CudaType {
    /// Scalar types.
    Float,
    Double,
    Int,
    UnsignedInt,
    LongLong,
    UnsignedLongLong,
    Bool,
    Void,

    /// Pointer types.
    Pointer {
        inner: Box<CudaType>,
        is_const: bool,
        restrict: bool,
    },

    /// Custom struct type.
    Struct(String),
}

impl CudaType {
    /// Convert to CUDA C type string.
    pub fn to_cuda_string(&self) -> String {
        match self {
            CudaType::Float => "float".to_string(),
            CudaType::Double => "double".to_string(),
            CudaType::Int => "int".to_string(),
            CudaType::UnsignedInt => "unsigned int".to_string(),
            CudaType::LongLong => "long long".to_string(),
            CudaType::UnsignedLongLong => "unsigned long long".to_string(),
            CudaType::Bool => "int".to_string(), // CUDA bool quirks
            CudaType::Void => "void".to_string(),
            CudaType::Pointer {
                inner,
                is_const,
                restrict,
            } => {
                let mut s = String::new();
                if *is_const {
                    s.push_str("const ");
                }
                s.push_str(&inner.to_cuda_string());
                s.push('*');
                if *restrict {
                    s.push_str(" __restrict__");
                }
                s
            }
            CudaType::Struct(name) => name.clone(),
        }
    }
}

/// Type mapper for Rust to CUDA conversions.
#[derive(Debug, Default)]
pub struct TypeMapper {
    /// Custom type mappings for user-defined structs.
    custom_types: std::collections::HashMap<String, CudaType>,
}

impl TypeMapper {
    /// Create a new type mapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a custom type mapping.
    pub fn register_type(&mut self, rust_name: &str, cuda_type: CudaType) {
        self.custom_types.insert(rust_name.to_string(), cuda_type);
    }

    /// Map a Rust type to CUDA.
    pub fn map_type(&self, ty: &Type) -> Result<CudaType> {
        match ty {
            Type::Path(path) => self.map_type_path(path),
            Type::Reference(reference) => self.map_reference(reference),
            Type::Tuple(tuple) if tuple.elems.is_empty() => Ok(CudaType::Void),
            _ => Err(TranspileError::Type(format!(
                "Unsupported type: {}",
                quote::quote!(#ty)
            ))),
        }
    }

    /// Map a type path (e.g., `f32`, `MyStruct`).
    fn map_type_path(&self, path: &TypePath) -> Result<CudaType> {
        let segments: Vec<_> = path.path.segments.iter().collect();

        if segments.len() != 1 {
            // Check for custom types first
            let full_path = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");

            if let Some(cuda_type) = self.custom_types.get(&full_path) {
                return Ok(cuda_type.clone());
            }

            return Err(TranspileError::Type(format!(
                "Complex path types not supported: {}",
                quote::quote!(#path)
            )));
        }

        let ident = &segments[0].ident;
        let type_name = ident.to_string();

        // Check primitive types
        match type_name.as_str() {
            "f32" => Ok(CudaType::Float),
            "f64" => Ok(CudaType::Double),
            "i32" => Ok(CudaType::Int),
            "u32" => Ok(CudaType::UnsignedInt),
            "i64" => Ok(CudaType::LongLong),
            "u64" => Ok(CudaType::UnsignedLongLong),
            "bool" => Ok(CudaType::Bool),
            "usize" => Ok(CudaType::UnsignedLongLong), // Assume 64-bit
            "isize" => Ok(CudaType::LongLong),
            // GridPos is a special marker type - we don't emit it
            "GridPos" => Ok(CudaType::Void),
            _ => {
                // Check custom types
                if let Some(cuda_type) = self.custom_types.get(&type_name) {
                    Ok(cuda_type.clone())
                } else {
                    // Assume it's a user struct
                    Ok(CudaType::Struct(type_name))
                }
            }
        }
    }

    /// Map a reference type (e.g., `&[f32]`, `&mut [f32]`).
    fn map_reference(&self, reference: &TypeReference) -> Result<CudaType> {
        let is_mutable = reference.mutability.is_some();

        match reference.elem.as_ref() {
            Type::Slice(slice) => {
                // &[T] -> const T* __restrict__
                // &mut [T] -> T* __restrict__
                let inner = self.map_type(&slice.elem)?;
                Ok(CudaType::Pointer {
                    inner: Box::new(inner),
                    is_const: !is_mutable,
                    restrict: true,
                })
            }
            Type::Path(path) => {
                // &T -> const T*
                // &mut T -> T*
                let inner = self.map_type_path(path)?;
                Ok(CudaType::Pointer {
                    inner: Box::new(inner),
                    is_const: !is_mutable,
                    restrict: false,
                })
            }
            _ => Err(TranspileError::Type(format!(
                "Unsupported reference type: {}",
                quote::quote!(#reference)
            ))),
        }
    }
}

/// Extract the inner element type from a slice reference.
pub fn get_slice_element_type(ty: &Type) -> Option<&Type> {
    if let Type::Reference(reference) = ty {
        if let Type::Slice(slice) = reference.elem.as_ref() {
            return Some(&slice.elem);
        }
    }
    None
}

/// Check if a type is a mutable reference.
pub fn is_mutable_reference(ty: &Type) -> bool {
    matches!(ty, Type::Reference(r) if r.mutability.is_some())
}

/// Check if a type is the GridPos context type.
pub fn is_grid_pos_type(ty: &Type) -> bool {
    if let Type::Path(path) = ty {
        if let Some(segment) = path.path.segments.last() {
            return segment.ident == "GridPos";
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_primitive_types() {
        let mapper = TypeMapper::new();

        let f32_ty: Type = parse_quote!(f32);
        assert_eq!(
            mapper.map_type(&f32_ty).unwrap().to_cuda_string(),
            "float"
        );

        let i32_ty: Type = parse_quote!(i32);
        assert_eq!(mapper.map_type(&i32_ty).unwrap().to_cuda_string(), "int");

        let bool_ty: Type = parse_quote!(bool);
        assert_eq!(mapper.map_type(&bool_ty).unwrap().to_cuda_string(), "int");
    }

    #[test]
    fn test_slice_types() {
        let mapper = TypeMapper::new();

        // &[f32] -> const float* __restrict__
        let slice_ty: Type = parse_quote!(&[f32]);
        assert_eq!(
            mapper.map_type(&slice_ty).unwrap().to_cuda_string(),
            "const float* __restrict__"
        );

        // &mut [f32] -> float* __restrict__
        let mut_slice_ty: Type = parse_quote!(&mut [f32]);
        assert_eq!(
            mapper.map_type(&mut_slice_ty).unwrap().to_cuda_string(),
            "float* __restrict__"
        );
    }

    #[test]
    fn test_grid_pos_type() {
        let ty: Type = parse_quote!(GridPos);
        assert!(is_grid_pos_type(&ty));

        let ty: Type = parse_quote!(f32);
        assert!(!is_grid_pos_type(&ty));
    }

    #[test]
    fn test_custom_types() {
        let mut mapper = TypeMapper::new();
        mapper.register_type("WaveParams", CudaType::Struct("WaveParams".to_string()));

        let ty: Type = parse_quote!(WaveParams);
        assert_eq!(
            mapper.map_type(&ty).unwrap().to_cuda_string(),
            "WaveParams"
        );
    }
}
