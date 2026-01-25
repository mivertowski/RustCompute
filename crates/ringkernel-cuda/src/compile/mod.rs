//! PTX compilation utilities for CUDA kernels.
//!
//! This module provides:
//! - File-based PTX caching to eliminate first-tick compilation overhead
//! - Source hashing for cache key generation
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::compile::PtxCache;
//!
//! let cache = PtxCache::new()?;
//! let source_hash = PtxCache::hash_source(cuda_source);
//!
//! // Try to get cached PTX
//! if let Some(ptx) = cache.get(&source_hash, "sm_89")? {
//!     return Ok(ptx);
//! }
//!
//! // Compile and cache
//! let ptx = ringkernel_cuda::compile_ptx(cuda_source)?;
//! cache.put(&source_hash, "sm_89", &ptx)?;
//! ```

mod cache;

pub use cache::{PtxCache, PtxCacheError, PtxCacheResult, PtxCacheStats, CACHE_VERSION};
