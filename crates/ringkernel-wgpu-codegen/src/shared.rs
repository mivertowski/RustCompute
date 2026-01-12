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

    /// Create a 3D shared memory declaration.
    pub fn new_3d(name: &str, element_type: WgslType, width: u32, height: u32, depth: u32) -> Self {
        Self {
            name: name.to_string(),
            element_type,
            dimensions: vec![width, height, depth],
        }
    }

    /// Total number of elements across all dimensions.
    pub fn total_elements(&self) -> u32 {
        self.dimensions.iter().product()
    }

    /// Generate WGSL declaration.
    ///
    /// Generates nested arrays for multi-dimensional shared memory:
    /// - 1D: `array<T, N>`
    /// - 2D: `array<array<T, W>, H>` (accessed as `arr[y][x]`)
    /// - 3D: `array<array<array<T, X>, Y>, Z>` (accessed as `arr[z][y][x]`)
    /// - 4D+: Linearized 1D array with accessor comment
    pub fn to_wgsl(&self) -> String {
        let type_str = self.element_type.to_wgsl();
        match self.dimensions.len() {
            0 => format!("var<workgroup> {}: {};", self.name, type_str),
            1 => format!(
                "var<workgroup> {}: array<{}, {}>;",
                self.name, type_str, self.dimensions[0]
            ),
            2 => {
                // 2D: array<array<T, W>, H> - [height][width] indexing (row-major)
                format!(
                    "var<workgroup> {}: array<array<{}, {}>, {}>;",
                    self.name, type_str, self.dimensions[0], self.dimensions[1]
                )
            }
            3 => {
                // 3D: array<array<array<T, X>, Y>, Z> - [depth][height][width] indexing
                format!(
                    "var<workgroup> {}: array<array<array<{}, {}>, {}>, {}>;",
                    self.name, type_str, self.dimensions[0], self.dimensions[1], self.dimensions[2]
                )
            }
            _ => {
                // 4D+ dimensions: linearize to 1D array
                // Include a comment showing the dimensions for reference
                let total = self.total_elements();
                let dims_str = self
                    .dimensions
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!(
                    "var<workgroup> {}: array<{}, {}>; // linearized {}D ({})",
                    self.name,
                    type_str,
                    total,
                    self.dimensions.len(),
                    dims_str
                )
            }
        }
    }

    /// Generate index calculation for linearized access to higher-dimensional arrays.
    ///
    /// For 4D+ arrays that are linearized, this generates the index formula.
    /// E.g., for dims [W, H, D, T]: `x + y * W + z * W * H + t * W * H * D`
    pub fn linearized_index_formula(&self, index_vars: &[&str]) -> Option<String> {
        if self.dimensions.len() < 4 || index_vars.len() != self.dimensions.len() {
            return None;
        }

        let mut terms = Vec::new();
        let mut stride = 1u32;

        for (i, var) in index_vars.iter().enumerate() {
            if stride == 1 {
                terms.push(var.to_string());
            } else {
                terms.push(format!("{} * {}u", var, stride));
            }
            stride *= self.dimensions[i];
        }

        Some(terms.join(" + "))
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

/// Marker type for 3D shared memory volumes.
///
/// This is a compile-time marker for volumetric shared memory (e.g., 3D stencil tiles).
/// The transpiler recognizes usage and generates appropriate WGSL workgroup variables.
/// Accessed as `volume[z][y][x]` in WGSL.
pub struct SharedVolume<T, const X: usize, const Y: usize, const Z: usize> {
    _marker: std::marker::PhantomData<T>,
}

impl<T, const X: usize, const Y: usize, const Z: usize> SharedVolume<T, X, Y, Z> {
    /// Get the volume width (X dimension).
    pub const fn width() -> usize {
        X
    }

    /// Get the volume height (Y dimension).
    pub const fn height() -> usize {
        Y
    }

    /// Get the volume depth (Z dimension).
    pub const fn depth() -> usize {
        Z
    }

    /// Total number of elements in the volume.
    pub const fn total() -> usize {
        X * Y * Z
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

    #[test]
    fn test_shared_memory_3d() {
        let decl = SharedMemoryDecl::new_3d("volume", WgslType::F32, 8, 8, 8);
        assert_eq!(
            decl.to_wgsl(),
            "var<workgroup> volume: array<array<array<f32, 8>, 8>, 8>;"
        );
        assert_eq!(decl.total_elements(), 512);
    }

    #[test]
    fn test_shared_memory_3d_asymmetric() {
        // 3D tile with halo: 10x10x10 for 8x8x8 interior with 1-cell halo
        let decl = SharedMemoryDecl::new_3d("tile_with_halo", WgslType::F32, 10, 10, 10);
        assert_eq!(
            decl.to_wgsl(),
            "var<workgroup> tile_with_halo: array<array<array<f32, 10>, 10>, 10>;"
        );
        assert_eq!(decl.total_elements(), 1000);
    }

    #[test]
    fn test_shared_memory_4d_linearized() {
        // 4D array gets linearized
        let decl = SharedMemoryDecl {
            name: "hypercube".to_string(),
            element_type: WgslType::F32,
            dimensions: vec![4, 4, 4, 4],
        };
        let wgsl = decl.to_wgsl();
        assert!(wgsl.contains("array<f32, 256>")); // 4*4*4*4 = 256
        assert!(wgsl.contains("linearized 4D"));
        assert!(wgsl.contains("4x4x4x4"));
    }

    #[test]
    fn test_linearized_index_formula() {
        let decl = SharedMemoryDecl {
            name: "data".to_string(),
            element_type: WgslType::F32,
            dimensions: vec![4, 8, 2, 3], // W=4, H=8, D=2, T=3
        };

        // Formula: x + y*4 + z*32 + t*64
        let formula = decl
            .linearized_index_formula(&["x", "y", "z", "t"])
            .unwrap();
        assert_eq!(formula, "x + y * 4u + z * 32u + t * 64u");
    }

    #[test]
    fn test_linearized_index_formula_returns_none_for_3d() {
        let decl = SharedMemoryDecl::new_3d("vol", WgslType::F32, 8, 8, 8);
        // 3D doesn't need linearization
        assert!(decl.linearized_index_formula(&["x", "y", "z"]).is_none());
    }

    #[test]
    fn test_shared_volume_marker() {
        // Verify the marker type constants work at compile time
        assert_eq!(SharedVolume::<f32, 8, 8, 8>::width(), 8);
        assert_eq!(SharedVolume::<f32, 8, 8, 8>::height(), 8);
        assert_eq!(SharedVolume::<f32, 8, 8, 8>::depth(), 8);
        assert_eq!(SharedVolume::<f32, 8, 8, 8>::total(), 512);

        // Asymmetric volume
        assert_eq!(SharedVolume::<f32, 16, 8, 4>::width(), 16);
        assert_eq!(SharedVolume::<f32, 16, 8, 4>::height(), 8);
        assert_eq!(SharedVolume::<f32, 16, 8, 4>::depth(), 4);
        assert_eq!(SharedVolume::<f32, 16, 8, 4>::total(), 512);
    }
}
