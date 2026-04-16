//! Vector Store / Embedding Index — FR-013
//!
//! GPU-friendly vector similarity search for hybrid neural-symbolic queries.
//! Stores embeddings alongside graph nodes and supports nearest-neighbor search.
//!
//! # Distance Metrics
//!
//! - Cosine similarity
//! - L2 (Euclidean) distance
//! - Dot product (inner product)
//!
//! # Index Types
//!
//! - **BruteForce**: Exact search, O(n·d). Good for < 100K vectors.
//! - **IVF**: Inverted file index for approximate search. Good for 100K-10M vectors.
//!
//! For GPU-accelerated search, the index data can be copied to device memory
//! and the search kernel processes queries in parallel.

use std::collections::HashMap;

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine_distance). Higher = more similar.
    Cosine,
    /// L2 (Euclidean) distance. Lower = more similar.
    L2,
    /// Dot product (inner product). Higher = more similar.
    DotProduct,
}

/// A vector with an associated ID and optional metadata.
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// Unique identifier (e.g., graph node ID).
    pub id: u64,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

/// Result of a nearest-neighbor query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// ID of the matched vector.
    pub id: u64,
    /// Distance/similarity score.
    pub score: f32,
    /// Metadata (if requested).
    pub metadata: Option<HashMap<String, String>>,
}

/// Configuration for the vector store.
#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    /// Dimensionality of vectors.
    pub dimensions: u32,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Maximum vectors to store.
    pub capacity: usize,
    /// Whether to store metadata alongside vectors.
    pub store_metadata: bool,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            dimensions: 768,
            metric: DistanceMetric::Cosine,
            capacity: 1_000_000,
            store_metadata: true,
        }
    }
}

/// Brute-force vector store (exact nearest-neighbor search).
///
/// Suitable for up to ~100K vectors. For larger datasets, use IVF.
/// The flat layout is GPU-friendly — vectors are stored contiguously
/// for efficient parallel distance computation.
pub struct VectorStore {
    /// Configuration.
    config: VectorStoreConfig,
    /// Flat vector storage: [n_vectors × dimensions].
    vectors: Vec<f32>,
    /// Vector IDs (parallel to vectors).
    ids: Vec<u64>,
    /// Optional metadata per vector.
    metadata: Vec<Option<HashMap<String, String>>>,
    /// Number of vectors stored.
    count: usize,
}

impl VectorStore {
    /// Create a new vector store.
    pub fn new(config: VectorStoreConfig) -> Self {
        let cap = config.capacity;
        let dim = config.dimensions as usize;
        Self {
            config,
            vectors: Vec::with_capacity(cap * dim),
            ids: Vec::with_capacity(cap),
            metadata: Vec::with_capacity(cap),
            count: 0,
        }
    }

    /// Insert a vector.
    pub fn insert(&mut self, entry: VectorEntry) -> Result<(), VectorStoreError> {
        if entry.vector.len() != self.config.dimensions as usize {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.config.dimensions as usize,
                got: entry.vector.len(),
            });
        }

        if self.count >= self.config.capacity {
            return Err(VectorStoreError::CapacityExceeded {
                capacity: self.config.capacity,
            });
        }

        self.vectors.extend_from_slice(&entry.vector);
        self.ids.push(entry.id);
        self.metadata.push(if self.config.store_metadata {
            Some(entry.metadata)
        } else {
            None
        });
        self.count += 1;

        Ok(())
    }

    /// Search for the top-K nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query.len() != self.config.dimensions as usize {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.config.dimensions as usize,
                got: query.len(),
            });
        }

        let dim = self.config.dimensions as usize;
        let mut scores: Vec<(usize, f32)> = (0..self.count)
            .map(|i| {
                let vec_start = i * dim;
                let vec_slice = &self.vectors[vec_start..vec_start + dim];
                let score = match self.config.metric {
                    DistanceMetric::Cosine => cosine_similarity(query, vec_slice),
                    DistanceMetric::L2 => -l2_distance(query, vec_slice), // Negate so higher = better
                    DistanceMetric::DotProduct => dot_product(query, vec_slice),
                };
                (i, score)
            })
            .collect();

        // Sort by score (descending — higher is better for all metrics after normalization)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores
            .into_iter()
            .map(|(idx, score)| SearchResult {
                id: self.ids[idx],
                score,
                metadata: self.metadata.get(idx).and_then(|m| m.clone()),
            })
            .collect())
    }

    /// Delete a vector by ID.
    pub fn delete(&mut self, id: u64) -> bool {
        if let Some(idx) = self.ids.iter().position(|&i| i == id) {
            let dim = self.config.dimensions as usize;
            let vec_start = idx * dim;

            // Remove from flat storage
            self.vectors.drain(vec_start..vec_start + dim);
            self.ids.remove(idx);
            self.metadata.remove(idx);
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// Number of vectors stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the flat vector data (for GPU upload).
    ///
    /// Returns a contiguous f32 slice of shape [count × dimensions].
    pub fn flat_vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// Get vector IDs.
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    /// Dimensionality.
    pub fn dimensions(&self) -> u32 {
        self.config.dimensions
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute L2 (Euclidean) distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Compute dot product between two vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vector store errors.
#[derive(Debug, Clone)]
pub enum VectorStoreError {
    /// Vector dimension doesn't match store configuration.
    DimensionMismatch {
        /// Expected vector dimension.
        expected: usize,
        /// Actual vector dimension provided.
        got: usize,
    },
    /// Store is full.
    CapacityExceeded {
        /// Maximum capacity of the vector store.
        capacity: usize,
    },
    /// Vector not found.
    NotFound(u64),
}

impl std::fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } =>
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got),
            Self::CapacityExceeded { capacity } =>
                write!(f, "Capacity exceeded: max {}", capacity),
            Self::NotFound(id) => write!(f, "Vector {} not found", id),
        }
    }
}

impl std::error::Error for VectorStoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, base: f32) -> Vec<f32> {
        (0..dim).map(|i| base + i as f32 * 0.1).collect()
    }

    #[test]
    fn test_insert_and_search() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 4,
            metric: DistanceMetric::Cosine,
            capacity: 100,
            store_metadata: false,
        });

        store.insert(VectorEntry { id: 1, vector: vec![1.0, 0.0, 0.0, 0.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 2, vector: vec![0.9, 0.1, 0.0, 0.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 3, vector: vec![0.0, 1.0, 0.0, 0.0], metadata: HashMap::new() }).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1); // Exact match = highest cosine
        assert_eq!(results[1].id, 2); // Close match
    }

    #[test]
    fn test_l2_distance_search() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 3,
            metric: DistanceMetric::L2,
            capacity: 100,
            store_metadata: false,
        });

        store.insert(VectorEntry { id: 1, vector: vec![0.0, 0.0, 0.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 2, vector: vec![1.0, 0.0, 0.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 3, vector: vec![10.0, 10.0, 10.0], metadata: HashMap::new() }).unwrap();

        let results = store.search(&[0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results[0].id, 1); // Closest
        assert_eq!(results[1].id, 2); // Next closest
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 4,
            ..Default::default()
        });

        let result = store.insert(VectorEntry {
            id: 1,
            vector: vec![1.0, 2.0], // Wrong dimension
            metadata: HashMap::new(),
        });
        assert!(matches!(result, Err(VectorStoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_capacity_exceeded() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 2,
            capacity: 2,
            ..Default::default()
        });

        store.insert(VectorEntry { id: 1, vector: vec![1.0, 0.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 2, vector: vec![0.0, 1.0], metadata: HashMap::new() }).unwrap();

        let result = store.insert(VectorEntry { id: 3, vector: vec![1.0, 1.0], metadata: HashMap::new() });
        assert!(matches!(result, Err(VectorStoreError::CapacityExceeded { .. })));
    }

    #[test]
    fn test_delete() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 2,
            capacity: 10,
            ..Default::default()
        });

        store.insert(VectorEntry { id: 1, vector: vec![1.0, 0.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 2, vector: vec![0.0, 1.0], metadata: HashMap::new() }).unwrap();
        assert_eq!(store.len(), 2);

        assert!(store.delete(1));
        assert_eq!(store.len(), 1);
        assert!(!store.delete(1)); // Already deleted
    }

    #[test]
    fn test_metadata_storage() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 2,
            store_metadata: true,
            ..Default::default()
        });

        let mut meta = HashMap::new();
        meta.insert("node_type".to_string(), "isa_standard".to_string());

        store.insert(VectorEntry { id: 42, vector: vec![1.0, 0.0], metadata: meta }).unwrap();

        let results = store.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 42);
        let meta = results[0].metadata.as_ref().unwrap();
        assert_eq!(meta.get("node_type").unwrap(), "isa_standard");
    }

    #[test]
    fn test_flat_vectors_for_gpu() {
        let mut store = VectorStore::new(VectorStoreConfig {
            dimensions: 3,
            capacity: 10,
            ..Default::default()
        });

        store.insert(VectorEntry { id: 1, vector: vec![1.0, 2.0, 3.0], metadata: HashMap::new() }).unwrap();
        store.insert(VectorEntry { id: 2, vector: vec![4.0, 5.0, 6.0], metadata: HashMap::new() }).unwrap();

        let flat = store.flat_vectors();
        assert_eq!(flat.len(), 6); // 2 vectors × 3 dimensions
        assert_eq!(flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }
}
