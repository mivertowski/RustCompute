//! IR type system.
//!
//! Defines types that can be used in GPU kernels across all backends.

use std::fmt;

/// Scalar types supported in IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// Boolean.
    Bool,
    /// 8-bit signed integer.
    I8,
    /// 16-bit signed integer.
    I16,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer.
    U16,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 16-bit floating point.
    F16,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point (not supported on all backends).
    F64,
}

impl ScalarType {
    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            ScalarType::Bool | ScalarType::I8 | ScalarType::U8 => 1,
            ScalarType::I16 | ScalarType::U16 | ScalarType::F16 => 2,
            ScalarType::I32 | ScalarType::U32 | ScalarType::F32 => 4,
            ScalarType::I64 | ScalarType::U64 | ScalarType::F64 => 8,
        }
    }

    /// Check if this is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::F16 | ScalarType::F32 | ScalarType::F64)
    }

    /// Check if this is a signed integer type.
    pub fn is_signed_int(&self) -> bool {
        matches!(
            self,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        )
    }

    /// Check if this is an unsigned integer type.
    pub fn is_unsigned_int(&self) -> bool {
        matches!(
            self,
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64
        )
    }

    /// Check if this is any integer type.
    pub fn is_int(&self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Check if this requires special capability (f64).
    pub fn requires_f64(&self) -> bool {
        matches!(self, ScalarType::F64)
    }

    /// Check if this requires 64-bit integer capability.
    pub fn requires_i64(&self) -> bool {
        matches!(self, ScalarType::I64 | ScalarType::U64)
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::F16 => write!(f, "f16"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
        }
    }
}

/// Vector types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorType {
    /// Element type.
    pub element: ScalarType,
    /// Number of elements (2, 3, or 4).
    pub count: u8,
}

impl VectorType {
    /// Create a new vector type.
    pub fn new(element: ScalarType, count: u8) -> Self {
        debug_assert!(count >= 2 && count <= 4, "Vector count must be 2, 3, or 4");
        Self { element, count }
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.element.size_bytes() * self.count as usize
    }
}

impl fmt::Display for VectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec{}<{}>", self.count, self.element)
    }
}

/// IR type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    /// Void type (for functions with no return).
    Void,
    /// Scalar type.
    Scalar(ScalarType),
    /// Vector type.
    Vector(VectorType),
    /// Pointer type.
    Ptr(Box<IrType>),
    /// Array type with static size.
    Array(Box<IrType>, usize),
    /// Slice type (runtime-sized array).
    Slice(Box<IrType>),
    /// Struct type with named fields.
    Struct(StructType),
    /// Function type.
    Function(FunctionType),
}

impl IrType {
    // Convenience constructors for common types

    /// Boolean type.
    pub const BOOL: IrType = IrType::Scalar(ScalarType::Bool);
    /// 32-bit signed integer.
    pub const I32: IrType = IrType::Scalar(ScalarType::I32);
    /// 64-bit signed integer.
    pub const I64: IrType = IrType::Scalar(ScalarType::I64);
    /// 32-bit unsigned integer.
    pub const U32: IrType = IrType::Scalar(ScalarType::U32);
    /// 64-bit unsigned integer.
    pub const U64: IrType = IrType::Scalar(ScalarType::U64);
    /// 32-bit float.
    pub const F32: IrType = IrType::Scalar(ScalarType::F32);
    /// 64-bit float.
    pub const F64: IrType = IrType::Scalar(ScalarType::F64);

    /// Create a pointer type.
    pub fn ptr(inner: IrType) -> Self {
        IrType::Ptr(Box::new(inner))
    }

    /// Create an array type.
    pub fn array(inner: IrType, size: usize) -> Self {
        IrType::Array(Box::new(inner), size)
    }

    /// Create a slice type.
    pub fn slice(inner: IrType) -> Self {
        IrType::Slice(Box::new(inner))
    }

    /// Get size in bytes (None for unsized types).
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            IrType::Void => Some(0),
            IrType::Scalar(s) => Some(s.size_bytes()),
            IrType::Vector(v) => Some(v.size_bytes()),
            IrType::Ptr(_) => Some(8), // 64-bit pointers
            IrType::Array(inner, count) => inner.size_bytes().map(|s| s * count),
            IrType::Slice(_) => None, // Unsized
            IrType::Struct(s) => s.size_bytes(),
            IrType::Function(_) => None,
        }
    }

    /// Check if this is a pointer type.
    pub fn is_ptr(&self) -> bool {
        matches!(self, IrType::Ptr(_))
    }

    /// Check if this is a scalar type.
    pub fn is_scalar(&self) -> bool {
        matches!(self, IrType::Scalar(_))
    }

    /// Check if this is a numeric type.
    pub fn is_numeric(&self) -> bool {
        match self {
            IrType::Scalar(s) => s.is_float() || s.is_int(),
            _ => false,
        }
    }

    /// Get the element type for pointers, arrays, and slices.
    pub fn element_type(&self) -> Option<&IrType> {
        match self {
            IrType::Ptr(inner) | IrType::Array(inner, _) | IrType::Slice(inner) => Some(inner),
            _ => None,
        }
    }
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => write!(f, "void"),
            IrType::Scalar(s) => write!(f, "{}", s),
            IrType::Vector(v) => write!(f, "{}", v),
            IrType::Ptr(inner) => write!(f, "*{}", inner),
            IrType::Array(inner, size) => write!(f, "[{}; {}]", inner, size),
            IrType::Slice(inner) => write!(f, "[{}]", inner),
            IrType::Struct(s) => write!(f, "struct {}", s.name),
            IrType::Function(ft) => write!(f, "{}", ft),
        }
    }
}

/// A struct type definition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    /// Struct name.
    pub name: String,
    /// Fields with names and types.
    pub fields: Vec<(String, IrType)>,
}

impl StructType {
    /// Create a new struct type.
    pub fn new(name: impl Into<String>, fields: Vec<(String, IrType)>) -> Self {
        Self {
            name: name.into(),
            fields,
        }
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> Option<usize> {
        let mut size = 0;
        for (_, ty) in &self.fields {
            size += ty.size_bytes()?;
        }
        Some(size)
    }

    /// Get field type by name.
    pub fn get_field(&self, name: &str) -> Option<&IrType> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ty)| ty)
    }

    /// Get field index by name.
    pub fn get_field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|(n, _)| n == name)
    }
}

/// A function type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    /// Parameter types.
    pub params: Vec<IrType>,
    /// Return type.
    pub return_type: Box<IrType>,
}

impl FunctionType {
    /// Create a new function type.
    pub fn new(params: Vec<IrType>, return_type: IrType) -> Self {
        Self {
            params,
            return_type: Box::new(return_type),
        }
    }
}

impl fmt::Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn(")?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param)?;
        }
        write!(f, ") -> {}", self.return_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_size() {
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
        assert_eq!(ScalarType::I32.size_bytes(), 4);
        assert_eq!(ScalarType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_scalar_classification() {
        assert!(ScalarType::F32.is_float());
        assert!(!ScalarType::I32.is_float());

        assert!(ScalarType::I32.is_signed_int());
        assert!(!ScalarType::U32.is_signed_int());

        assert!(ScalarType::U32.is_unsigned_int());
        assert!(!ScalarType::I32.is_unsigned_int());
    }

    #[test]
    fn test_vector_type() {
        let v = VectorType::new(ScalarType::F32, 4);
        assert_eq!(v.size_bytes(), 16);
        assert_eq!(format!("{}", v), "vec4<f32>");
    }

    #[test]
    fn test_ir_type_display() {
        assert_eq!(format!("{}", IrType::I32), "i32");
        assert_eq!(format!("{}", IrType::ptr(IrType::F32)), "*f32");
        assert_eq!(format!("{}", IrType::array(IrType::I32, 16)), "[i32; 16]");
    }

    #[test]
    fn test_struct_type() {
        let s = StructType::new(
            "Point",
            vec![
                ("x".to_string(), IrType::F32),
                ("y".to_string(), IrType::F32),
            ],
        );
        assert_eq!(s.size_bytes(), Some(8));
        assert_eq!(s.get_field("x"), Some(&IrType::F32));
        assert_eq!(s.get_field_index("y"), Some(1));
    }

    #[test]
    fn test_function_type() {
        let ft = FunctionType::new(vec![IrType::I32, IrType::F32], IrType::F32);
        assert_eq!(format!("{}", ft), "fn(i32, f32) -> f32");
    }
}
