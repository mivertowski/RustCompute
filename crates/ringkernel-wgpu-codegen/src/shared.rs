//! Shared (workgroup) memory support for WGSL code generation.
//!
//! Maps Rust shared memory patterns to WGSL workgroup variables.

use crate::types::WgslType;

/// Declaration of a shared memory variable.
#[derive(Debug, Clone)]
pub struct SharedMemoryDecl {
    /// Variable name.
    pub name: String,
    /// Element type.
    pub element_type: WgslType,
    /// Dimensions (1D, 2D, etc.).
    pub dimensions: Vec<u32>,
}

impl SharedMemoryDecl {
    /// Create a 1D shared memory declaration.
    pub fn new_1d(name: &str, element_type: WgslType, size: u32) -> Self {
        Self {
            name: name.to_string(),
            element_type,
            dimensions: vec![size],
        }
    }

    /// Create a 2D shared memory declaration.
    pub fn new_2d(name: &str, element_type: WgslType, width: u32, height: u32) -> Self {
        Self {
            name: name.to_string(),
            element_type,
            dimensions: vec![width, height],
        }
    }

    /// Generate WGSL declaration.
    pub fn to_wgsl(&self) -> String {
        let type_str = self.element_type.to_wgsl();
        match self.dimensions.len() {
            1 => format!(
                "var<workgroup> {}: array<{}, {}>;",
                self.name, type_str, self.dimensions[0]
            ),
            2 => format!(
                "var<workgroup> {}: array<array<{}, {}>, {}>;",
                self.name, type_str, self.dimensions[0], self.dimensions[1]
            ),
            _ => format!(
                "var<workgroup> {}: array<{}, {}>; // TODO: higher dimensions",
                self.name, type_str, self.dimensions[0]
            ),
        }
    }
}

/// Configuration for shared memory in a kernel.
#[derive(Debug, Clone, Default)]
pub struct SharedMemoryConfig {
    /// List of shared memory declarations.
    pub declarations: Vec<SharedMemoryDecl>,
}

impl SharedMemoryConfig {
    /// Create a new empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a shared memory declaration.
    pub fn add(&mut self, decl: SharedMemoryDecl) {
        self.declarations.push(decl);
    }

    /// Generate all WGSL declarations.
    pub fn to_wgsl(&self) -> String {
        self.declarations
            .iter()
            .map(|d| d.to_wgsl())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Marker type for 2D shared memory tiles.
///
/// This is a compile-time marker. The transpiler recognizes usage and
/// generates appropriate WGSL workgroup variables.
pub struct SharedTile<T, const W: usize, const H: usize> {
    _marker: std::marker::PhantomData<T>,
}

impl<T, const W: usize, const H: usize> SharedTile<T, W, H> {
    /// Get the tile width.
    pub const fn width() -> usize {
        W
    }

    /// Get the tile height.
    pub const fn height() -> usize {
        H
    }
}

/// Marker type for 1D shared memory arrays.
pub struct SharedArray<T, const N: usize> {
    _marker: std::marker::PhantomData<T>,
}

impl<T, const N: usize> SharedArray<T, N> {
    /// Get the array size.
    pub const fn size() -> usize {
        N
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_memory_1d() {
        let decl = SharedMemoryDecl::new_1d("cache", WgslType::F32, 256);
        assert_eq!(decl.to_wgsl(), "var<workgroup> cache: array<f32, 256>;");
    }

    #[test]
    fn test_shared_memory_2d() {
        let decl = SharedMemoryDecl::new_2d("tile", WgslType::F32, 16, 16);
        assert_eq!(
            decl.to_wgsl(),
            "var<workgroup> tile: array<array<f32, 16>, 16>;"
        );
    }

    #[test]
    fn test_shared_memory_config() {
        let mut config = SharedMemoryConfig::new();
        config.add(SharedMemoryDecl::new_1d("a", WgslType::I32, 64));
        config.add(SharedMemoryDecl::new_1d("b", WgslType::F32, 128));

        let wgsl = config.to_wgsl();
        assert!(wgsl.contains("var<workgroup> a: array<i32, 64>;"));
        assert!(wgsl.contains("var<workgroup> b: array<f32, 128>;"));
    }
}
