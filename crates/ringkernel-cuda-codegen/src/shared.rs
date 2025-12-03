//! Shared memory support for CUDA code generation.
//!
//! This module provides types and utilities for working with CUDA shared memory
//! (`__shared__`) in the Rust DSL. Shared memory is fast on-chip memory that is
//! shared among all threads in a block.
//!
//! # Overview
//!
//! Shared memory is crucial for efficient GPU programming:
//! - Much faster than global memory (~100x lower latency)
//! - Shared among all threads in a block
//! - Limited size (typically 48KB-164KB per SM)
//! - Requires explicit synchronization after writes
//!
//! # Usage in DSL
//!
//! ```ignore
//! use ringkernel_cuda_codegen::shared::SharedTile;
//!
//! fn kernel(data: &[f32], out: &mut [f32], width: i32) {
//!     // Declare a 16x16 shared memory tile
//!     let tile = SharedTile::<f32, 16, 16>::new();
//!
//!     // Load from global memory
//!     let gx = block_idx_x() * 16 + thread_idx_x();
//!     let gy = block_idx_y() * 16 + thread_idx_y();
//!     tile.set(thread_idx_x(), thread_idx_y(), data[gy * width + gx]);
//!
//!     // Synchronize before reading
//!     sync_threads();
//!
//!     // Read from shared memory
//!     let val = tile.get(thread_idx_x(), thread_idx_y());
//!     out[gy * width + gx] = val * 2.0;
//! }
//! ```
//!
//! # Generated CUDA
//!
//! The above DSL generates:
//!
//! ```cuda
//! __shared__ float tile[16][16];
//! int gx = blockIdx.x * 16 + threadIdx.x;
//! int gy = blockIdx.y * 16 + threadIdx.y;
//! tile[threadIdx.y][threadIdx.x] = data[gy * width + gx];
//! __syncthreads();
//! float val = tile[threadIdx.y][threadIdx.x];
//! out[gy * width + gx] = val * 2.0f;
//! ```

use std::marker::PhantomData;

/// A 2D shared memory tile.
///
/// This type represents a 2D array in CUDA shared memory. On the CPU side,
/// it's a zero-sized type that serves as a marker for the transpiler.
///
/// # Type Parameters
///
/// * `T` - Element type (f32, f64, i32, etc.)
/// * `W` - Tile width (columns)
/// * `H` - Tile height (rows)
///
/// # Example
///
/// ```ignore
/// // 16x16 tile of floats
/// let tile = SharedTile::<f32, 16, 16>::new();
///
/// // 32x8 tile for matrix operations
/// let mat_tile = SharedTile::<f32, 32, 8>::new();
/// ```
#[derive(Debug)]
pub struct SharedTile<T, const W: usize, const H: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Default + Copy, const W: usize, const H: usize> SharedTile<T, W, H> {
    /// Create a new shared memory tile.
    ///
    /// On GPU, this translates to: `__shared__ T tile[H][W];`
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the tile width.
    #[inline]
    pub const fn width() -> usize {
        W
    }

    /// Get the tile height.
    #[inline]
    pub const fn height() -> usize {
        H
    }

    /// Get the total number of elements.
    #[inline]
    pub const fn size() -> usize {
        W * H
    }

    /// Get an element from the tile (CPU stub - actual access on GPU).
    ///
    /// On GPU, this translates to: `tile[y][x]`
    ///
    /// # Arguments
    ///
    /// * `x` - Column index (0..W)
    /// * `y` - Row index (0..H)
    #[inline]
    pub fn get(&self, _x: i32, _y: i32) -> T {
        // This is a transpiler marker - actual access happens on GPU
        T::default()
    }

    /// Set an element in the tile (CPU stub - actual write on GPU).
    ///
    /// On GPU, this translates to: `tile[y][x] = value;`
    ///
    /// # Arguments
    ///
    /// * `x` - Column index (0..W)
    /// * `y` - Row index (0..H)
    /// * `value` - Value to store
    #[inline]
    pub fn set(&mut self, _x: i32, _y: i32, _value: T) {
        // This is a transpiler marker - actual write happens on GPU
    }
}

impl<T: Default + Copy, const W: usize, const H: usize> Default for SharedTile<T, W, H> {
    fn default() -> Self {
        Self::new()
    }
}

/// A 1D shared memory array.
///
/// Simpler than `SharedTile` for linear data access patterns.
///
/// # Type Parameters
///
/// * `T` - Element type
/// * `N` - Array size
#[derive(Debug)]
pub struct SharedArray<T, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Default + Copy, const N: usize> SharedArray<T, N> {
    /// Create a new shared memory array.
    ///
    /// On GPU, this translates to: `__shared__ T arr[N];`
    #[inline]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the array size.
    #[inline]
    pub const fn size() -> usize {
        N
    }

    /// Get an element from the array (CPU stub - actual access on GPU).
    #[inline]
    pub fn get(&self, _idx: i32) -> T {
        T::default()
    }

    /// Set an element in the array (CPU stub - actual write on GPU).
    #[inline]
    pub fn set(&mut self, _idx: i32, _value: T) {
        // This is a transpiler marker - actual write happens on GPU
    }
}

impl<T: Default + Copy, const N: usize> Default for SharedArray<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a shared memory declaration for transpilation.
#[derive(Debug, Clone)]
pub struct SharedMemoryDecl {
    /// Variable name.
    pub name: String,
    /// Element type (CUDA type string).
    pub element_type: String,
    /// Dimensions (1D: [size], 2D: [height, width]).
    pub dimensions: Vec<usize>,
}

impl SharedMemoryDecl {
    /// Create a 1D shared memory declaration.
    pub fn array(name: impl Into<String>, element_type: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            element_type: element_type.into(),
            dimensions: vec![size],
        }
    }

    /// Create a 2D shared memory declaration.
    pub fn tile(
        name: impl Into<String>,
        element_type: impl Into<String>,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            name: name.into(),
            element_type: element_type.into(),
            dimensions: vec![height, width], // Row-major: [rows][cols]
        }
    }

    /// Generate CUDA declaration string.
    ///
    /// # Returns
    ///
    /// A string like `__shared__ float tile[16][16];`
    pub fn to_cuda_decl(&self) -> String {
        let dims: String = self
            .dimensions
            .iter()
            .map(|d| format!("[{}]", d))
            .collect();

        format!("__shared__ {} {}{};", self.element_type, self.name, dims)
    }

    /// Generate CUDA access expression.
    ///
    /// # Arguments
    ///
    /// * `indices` - Index expressions for each dimension
    ///
    /// # Returns
    ///
    /// A string like `tile[y][x]`
    pub fn to_cuda_access(&self, indices: &[String]) -> String {
        let idx_str: String = indices.iter().map(|i| format!("[{}]", i)).collect();
        format!("{}{}", self.name, idx_str)
    }
}

/// Shared memory configuration for a kernel.
#[derive(Debug, Clone, Default)]
pub struct SharedMemoryConfig {
    /// All shared memory declarations in the kernel.
    pub declarations: Vec<SharedMemoryDecl>,
}

impl SharedMemoryConfig {
    /// Create a new empty configuration.
    pub fn new() -> Self {
        Self {
            declarations: Vec::new(),
        }
    }

    /// Add a shared memory declaration.
    pub fn add(&mut self, decl: SharedMemoryDecl) {
        self.declarations.push(decl);
    }

    /// Add a 1D shared array.
    pub fn add_array(
        &mut self,
        name: impl Into<String>,
        element_type: impl Into<String>,
        size: usize,
    ) {
        self.declarations
            .push(SharedMemoryDecl::array(name, element_type, size));
    }

    /// Add a 2D shared tile.
    pub fn add_tile(
        &mut self,
        name: impl Into<String>,
        element_type: impl Into<String>,
        width: usize,
        height: usize,
    ) {
        self.declarations
            .push(SharedMemoryDecl::tile(name, element_type, width, height));
    }

    /// Generate all CUDA shared memory declarations.
    pub fn generate_declarations(&self, indent: &str) -> String {
        self.declarations
            .iter()
            .map(|d| format!("{}{}", indent, d.to_cuda_decl()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check if any shared memory is used.
    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }

    /// Calculate total shared memory size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.declarations
            .iter()
            .map(|d| {
                let elem_size = match d.element_type.as_str() {
                    "float" => 4,
                    "double" => 8,
                    "int" => 4,
                    "unsigned int" => 4,
                    "long long" | "unsigned long long" => 8,
                    "short" | "unsigned short" => 2,
                    "char" | "unsigned char" => 1,
                    _ => 4, // Default assumption
                };
                let count: usize = d.dimensions.iter().product();
                elem_size * count
            })
            .sum()
    }
}

/// Parse a SharedTile type to extract dimensions.
///
/// # Arguments
///
/// * `type_path` - The type path (e.g., `SharedTile::<f32, 16, 16>`)
///
/// # Returns
///
/// `(element_type, width, height)` if successfully parsed.
pub fn parse_shared_tile_type(type_str: &str) -> Option<(String, usize, usize)> {
    // Pattern: SharedTile<T, W, H> or SharedTile::<T, W, H>
    let inner = type_str
        .strip_prefix("SharedTile")?
        .trim_start_matches("::")
        .strip_prefix('<')?
        .strip_suffix('>')?;

    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() != 3 {
        return None;
    }

    let element_type = parts[0].to_string();
    let width: usize = parts[1].parse().ok()?;
    let height: usize = parts[2].parse().ok()?;

    Some((element_type, width, height))
}

/// Parse a SharedArray type to extract size.
///
/// # Arguments
///
/// * `type_str` - The type path (e.g., `SharedArray::<f32, 256>`)
///
/// # Returns
///
/// `(element_type, size)` if successfully parsed.
pub fn parse_shared_array_type(type_str: &str) -> Option<(String, usize)> {
    // Pattern: SharedArray<T, N> or SharedArray::<T, N>
    let inner = type_str
        .strip_prefix("SharedArray")?
        .trim_start_matches("::")
        .strip_prefix('<')?
        .strip_suffix('>')?;

    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        return None;
    }

    let element_type = parts[0].to_string();
    let size: usize = parts[1].parse().ok()?;

    Some((element_type, size))
}

/// Map Rust element type to CUDA type for shared memory.
pub fn rust_to_cuda_element_type(rust_type: &str) -> &'static str {
    match rust_type {
        "f32" => "float",
        "f64" => "double",
        "i32" => "int",
        "u32" => "unsigned int",
        "i64" => "long long",
        "u64" => "unsigned long long",
        "i16" => "short",
        "u16" => "unsigned short",
        "i8" => "char",
        "u8" => "unsigned char",
        "bool" => "int",
        _ => "float", // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_tile_dimensions() {
        assert_eq!(SharedTile::<f32, 16, 16>::width(), 16);
        assert_eq!(SharedTile::<f32, 16, 16>::height(), 16);
        assert_eq!(SharedTile::<f32, 16, 16>::size(), 256);

        assert_eq!(SharedTile::<f32, 32, 8>::width(), 32);
        assert_eq!(SharedTile::<f32, 32, 8>::height(), 8);
        assert_eq!(SharedTile::<f32, 32, 8>::size(), 256);
    }

    #[test]
    fn test_shared_array_size() {
        assert_eq!(SharedArray::<f32, 256>::size(), 256);
        assert_eq!(SharedArray::<i32, 1024>::size(), 1024);
    }

    #[test]
    fn test_shared_memory_decl_1d() {
        let decl = SharedMemoryDecl::array("buffer", "float", 256);
        assert_eq!(decl.to_cuda_decl(), "__shared__ float buffer[256];");
        assert_eq!(decl.to_cuda_access(&["i".to_string()]), "buffer[i]");
    }

    #[test]
    fn test_shared_memory_decl_2d() {
        let decl = SharedMemoryDecl::tile("tile", "float", 16, 16);
        assert_eq!(decl.to_cuda_decl(), "__shared__ float tile[16][16];");
        assert_eq!(
            decl.to_cuda_access(&["y".to_string(), "x".to_string()]),
            "tile[y][x]"
        );
    }

    #[test]
    fn test_shared_memory_config() {
        let mut config = SharedMemoryConfig::new();
        config.add_tile("tile", "float", 16, 16);
        config.add_array("temp", "int", 128);

        let decls = config.generate_declarations("    ");
        assert!(decls.contains("__shared__ float tile[16][16];"));
        assert!(decls.contains("__shared__ int temp[128];"));
    }

    #[test]
    fn test_total_bytes() {
        let mut config = SharedMemoryConfig::new();
        config.add_tile("tile", "float", 16, 16); // 16*16*4 = 1024
        config.add_array("temp", "double", 64); // 64*8 = 512

        assert_eq!(config.total_bytes(), 1024 + 512);
    }

    #[test]
    fn test_parse_shared_tile_type() {
        let result = parse_shared_tile_type("SharedTile::<f32, 16, 16>");
        assert_eq!(result, Some(("f32".to_string(), 16, 16)));

        let result2 = parse_shared_tile_type("SharedTile<i32, 32, 8>");
        assert_eq!(result2, Some(("i32".to_string(), 32, 8)));
    }

    #[test]
    fn test_parse_shared_array_type() {
        let result = parse_shared_array_type("SharedArray::<f32, 256>");
        assert_eq!(result, Some(("f32".to_string(), 256)));

        let result2 = parse_shared_array_type("SharedArray<u32, 1024>");
        assert_eq!(result2, Some(("u32".to_string(), 1024)));
    }

    #[test]
    fn test_rust_to_cuda_element_type() {
        assert_eq!(rust_to_cuda_element_type("f32"), "float");
        assert_eq!(rust_to_cuda_element_type("f64"), "double");
        assert_eq!(rust_to_cuda_element_type("i32"), "int");
        assert_eq!(rust_to_cuda_element_type("u64"), "unsigned long long");
    }
}
