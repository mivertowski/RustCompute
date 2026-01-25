//! PTX caching for GPU kernel compilation.
//!
//! This module provides file-based caching of compiled PTX code to eliminate
//! first-tick kernel compilation overhead. On first use, kernels are compiled
//! via NVRTC and cached to disk. Subsequent runs load the cached PTX directly,
//! reducing first-tick latency from 11-32ms to <1ms.
//!
//! # Cache Key Generation
//!
//! Cache keys are generated from:
//! - SHA-256 hash of the CUDA source code
//! - CUDA compute capability (e.g., sm_86)
//! - Cache format version (for invalidation on format changes)
//!
//! # Cache Location
//!
//! Default: `~/.cache/ringkernel/ptx/`
//! Override: `RINGKERNEL_PTX_CACHE_DIR` environment variable

use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Current cache format version. Increment to invalidate all cached entries.
pub const CACHE_VERSION: u32 = 1;

/// Error types for PTX cache operations.
#[derive(Debug, thiserror::Error)]
pub enum PtxCacheError {
    /// Failed to create cache directory.
    #[error("Failed to create cache directory: {0}")]
    CreateDirFailed(std::io::Error),

    /// Failed to read cached PTX.
    #[error("Failed to read cached PTX: {0}")]
    ReadFailed(std::io::Error),

    /// Failed to write PTX to cache.
    #[error("Failed to write PTX to cache: {0}")]
    WriteFailed(std::io::Error),

    /// Cache entry is corrupted or invalid.
    #[error("Cache entry corrupted: {0}")]
    Corrupted(String),
}

/// Result type for PTX cache operations.
pub type PtxCacheResult<T> = Result<T, PtxCacheError>;

/// File-based PTX cache for eliminating kernel compilation overhead.
///
/// # Example
///
/// ```ignore
/// use ringkernel_cuda::compile::PtxCache;
///
/// let cache = PtxCache::new()?;
/// let source_hash = PtxCache::hash_source(cuda_source);
///
/// // Try to get cached PTX
/// if let Some(ptx) = cache.get(&source_hash, "sm_86")? {
///     return Ok(ptx);
/// }
///
/// // Compile and cache
/// let ptx = compile_with_nvrtc(cuda_source)?;
/// cache.put(&source_hash, "sm_86", &ptx)?;
/// ```
#[derive(Debug, Clone)]
pub struct PtxCache {
    /// Directory where cached PTX files are stored.
    cache_dir: PathBuf,
    /// Whether caching is enabled.
    enabled: bool,
}

impl Default for PtxCache {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            cache_dir: PathBuf::new(),
            enabled: false,
        })
    }
}

impl PtxCache {
    /// Creates a new PTX cache using the default or environment-configured directory.
    ///
    /// Default location: `~/.cache/ringkernel/ptx/`
    /// Override with: `RINGKERNEL_PTX_CACHE_DIR` environment variable
    pub fn new() -> PtxCacheResult<Self> {
        let cache_dir = Self::default_cache_dir();
        Self::with_dir(cache_dir)
    }

    /// Creates a PTX cache with a specific directory.
    pub fn with_dir(cache_dir: PathBuf) -> PtxCacheResult<Self> {
        // Ensure cache directory exists
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).map_err(PtxCacheError::CreateDirFailed)?;
        }

        Ok(Self {
            cache_dir,
            enabled: true,
        })
    }

    /// Creates a disabled cache (no-op for testing or when disk access is unavailable).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            cache_dir: PathBuf::new(),
            enabled: false,
        }
    }

    /// Returns whether caching is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the cache directory path.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Computes the SHA-256 hash of CUDA source code.
    ///
    /// The hash is returned as a lowercase hex string.
    #[must_use]
    pub fn hash_source(source: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(source.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)
    }

    /// Generates the cache file path for a given source hash and compute capability.
    fn cache_path(&self, source_hash: &str, compute_cap: &str) -> PathBuf {
        let filename = format!("{}_{}_{}.ptx", source_hash, compute_cap, CACHE_VERSION);
        self.cache_dir.join(filename)
    }

    /// Attempts to retrieve cached PTX for the given source hash and compute capability.
    ///
    /// Returns `None` if no cache entry exists or the entry is invalid.
    pub fn get(&self, source_hash: &str, compute_cap: &str) -> PtxCacheResult<Option<String>> {
        if !self.enabled {
            return Ok(None);
        }

        let path = self.cache_path(source_hash, compute_cap);

        if !path.exists() {
            return Ok(None);
        }

        let mut file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(PtxCacheError::ReadFailed(e)),
        };

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(PtxCacheError::ReadFailed)?;

        // Validate PTX header
        if !contents.starts_with("//") && !contents.contains(".version") {
            // Invalid PTX, remove corrupted cache entry
            let _ = fs::remove_file(&path);
            return Err(PtxCacheError::Corrupted("Invalid PTX header".to_string()));
        }

        tracing::debug!(
            path = %path.display(),
            size = contents.len(),
            "Loaded cached PTX"
        );

        Ok(Some(contents))
    }

    /// Stores compiled PTX in the cache.
    ///
    /// Creates the cache file atomically using a temporary file and rename.
    pub fn put(&self, source_hash: &str, compute_cap: &str, ptx: &str) -> PtxCacheResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let path = self.cache_path(source_hash, compute_cap);
        let temp_path = path.with_extension("ptx.tmp");

        // Write to temporary file
        let mut file = fs::File::create(&temp_path).map_err(PtxCacheError::WriteFailed)?;
        file.write_all(ptx.as_bytes())
            .map_err(PtxCacheError::WriteFailed)?;
        file.sync_all().map_err(PtxCacheError::WriteFailed)?;
        drop(file);

        // Atomically rename to final path
        fs::rename(&temp_path, &path).map_err(PtxCacheError::WriteFailed)?;

        tracing::debug!(
            path = %path.display(),
            size = ptx.len(),
            "Cached PTX"
        );

        Ok(())
    }

    /// Clears all cached PTX files.
    pub fn clear(&self) -> PtxCacheResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let entries = fs::read_dir(&self.cache_dir).map_err(PtxCacheError::ReadFailed)?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "ptx") {
                let _ = fs::remove_file(path);
            }
        }

        Ok(())
    }

    /// Returns statistics about the cache.
    #[must_use]
    pub fn stats(&self) -> PtxCacheStats {
        if !self.enabled {
            return PtxCacheStats::default();
        }

        let mut total_entries = 0;
        let mut total_bytes = 0;

        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "ptx") {
                    total_entries += 1;
                    if let Ok(metadata) = fs::metadata(&path) {
                        total_bytes += metadata.len();
                    }
                }
            }
        }

        PtxCacheStats {
            entries: total_entries,
            bytes: total_bytes,
            cache_dir: self.cache_dir.clone(),
        }
    }

    /// Returns the default cache directory.
    fn default_cache_dir() -> PathBuf {
        // Check environment variable first
        if let Ok(dir) = std::env::var("RINGKERNEL_PTX_CACHE_DIR") {
            return PathBuf::from(dir);
        }

        // Use platform-specific cache directory
        #[cfg(target_os = "linux")]
        {
            if let Ok(home) = std::env::var("HOME") {
                return PathBuf::from(home).join(".cache/ringkernel/ptx");
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(home) = std::env::var("HOME") {
                return PathBuf::from(home).join("Library/Caches/ringkernel/ptx");
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
                return PathBuf::from(local_app_data).join("ringkernel/ptx");
            }
        }

        // Fallback to temp directory
        std::env::temp_dir().join("ringkernel/ptx")
    }

    /// Compiles CUDA source to PTX with caching.
    ///
    /// This is a convenience method that combines hashing, cache lookup,
    /// compilation, and cache storage.
    ///
    /// # Arguments
    ///
    /// * `source` - CUDA C source code
    /// * `compute_cap` - Compute capability string (e.g., "sm_89")
    ///
    /// # Returns
    ///
    /// The compiled PTX code.
    #[cfg(feature = "cuda")]
    pub fn compile_cached(
        &self,
        source: &str,
        compute_cap: &str,
    ) -> ringkernel_core::error::Result<String> {
        let hash = Self::hash_source(source);

        // Try cache first
        if let Ok(Some(cached)) = self.get(&hash, compute_cap) {
            tracing::debug!("PTX cache hit for {}", &hash[..8]);
            return Ok(cached);
        }

        // Compile with NVRTC
        tracing::debug!("PTX cache miss for {}, compiling...", &hash[..8]);
        let ptx = crate::compile_ptx(source)?;

        // Cache the result
        if let Err(e) = self.put(&hash, compute_cap, &ptx) {
            tracing::warn!("Failed to cache PTX: {}", e);
        }

        Ok(ptx)
    }
}

/// Statistics about the PTX cache.
#[derive(Debug, Clone, Default)]
pub struct PtxCacheStats {
    /// Number of cached PTX entries.
    pub entries: usize,
    /// Total size of cached PTX in bytes.
    pub bytes: u64,
    /// Cache directory path.
    pub cache_dir: PathBuf,
}

impl std::fmt::Display for PtxCacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PTX Cache: {} entries, {:.2} KB at {}",
            self.entries,
            self.bytes as f64 / 1024.0,
            self.cache_dir.display()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_hash_source() {
        let source = "extern \"C\" __global__ void test() {}";
        let hash = PtxCache::hash_source(source);

        // Hash should be 64 hex characters (256 bits)
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

        // Same source should produce same hash
        let hash2 = PtxCache::hash_source(source);
        assert_eq!(hash, hash2);

        // Different source should produce different hash
        let hash3 = PtxCache::hash_source("different source");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_cache_miss() {
        let temp_dir = TempDir::new().unwrap();
        let cache = PtxCache::with_dir(temp_dir.path().to_path_buf()).unwrap();

        let result = cache.get("nonexistent_hash", "sm_86").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_put_get() {
        let temp_dir = TempDir::new().unwrap();
        let cache = PtxCache::with_dir(temp_dir.path().to_path_buf()).unwrap();

        let source = "extern \"C\" __global__ void test() {}";
        let hash = PtxCache::hash_source(source);
        let ptx = "// Generated PTX\n.version 7.5\n.target sm_86";

        // Put PTX in cache
        cache.put(&hash, "sm_86", ptx).unwrap();

        // Get PTX from cache
        let cached = cache.get(&hash, "sm_86").unwrap();
        assert_eq!(cached, Some(ptx.to_string()));
    }

    #[test]
    fn test_cache_version_isolation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = PtxCache::with_dir(temp_dir.path().to_path_buf()).unwrap();

        let hash = "testhash123";
        let ptx = "// Generated PTX\n.version 7.5";

        cache.put(hash, "sm_86", ptx).unwrap();

        // Different compute capability should not match
        let cached = cache.get(hash, "sm_75").unwrap();
        assert!(cached.is_none());
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache = PtxCache::with_dir(temp_dir.path().to_path_buf()).unwrap();

        // Initially empty
        let stats = cache.stats();
        assert_eq!(stats.entries, 0);

        // Add some entries
        cache
            .put("hash1", "sm_86", "// ptx1\n.version 7.5")
            .unwrap();
        cache
            .put("hash2", "sm_86", "// ptx2\n.version 7.5")
            .unwrap();

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert!(stats.bytes > 0);
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let cache = PtxCache::with_dir(temp_dir.path().to_path_buf()).unwrap();

        cache
            .put("hash1", "sm_86", "// ptx1\n.version 7.5")
            .unwrap();
        cache
            .put("hash2", "sm_86", "// ptx2\n.version 7.5")
            .unwrap();

        assert_eq!(cache.stats().entries, 2);

        cache.clear().unwrap();

        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn test_disabled_cache() {
        let cache = PtxCache::disabled();

        assert!(!cache.is_enabled());

        // Operations should no-op
        cache.put("hash", "sm_86", "ptx").unwrap();
        let result = cache.get("hash", "sm_86").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_stats_display() {
        let stats = PtxCacheStats {
            entries: 5,
            bytes: 10240,
            cache_dir: PathBuf::from("/tmp/test"),
        };
        let display = format!("{}", stats);
        assert!(display.contains("5 entries"));
        assert!(display.contains("10.00 KB"));
    }
}
