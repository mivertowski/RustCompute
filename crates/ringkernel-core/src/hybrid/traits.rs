//! Traits for hybrid CPU-GPU workloads.

use super::error::HybridResult;

/// Trait for workloads that can be executed on CPU or GPU.
///
/// Implementors provide both CPU and GPU execution paths, allowing the
/// `HybridDispatcher` to choose the optimal backend based on workload size
/// and runtime measurements.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::hybrid::{HybridWorkload, HybridResult};
///
/// struct VectorAdd {
///     a: Vec<f32>,
///     b: Vec<f32>,
/// }
///
/// impl HybridWorkload for VectorAdd {
///     type Result = Vec<f32>;
///
///     fn workload_size(&self) -> usize {
///         self.a.len()
///     }
///
///     fn execute_cpu(&self) -> Self::Result {
///         self.a.iter().zip(&self.b).map(|(a, b)| a + b).collect()
///     }
///
///     fn execute_gpu(&self) -> HybridResult<Self::Result> {
///         // GPU implementation
///         todo!("GPU kernel execution")
///     }
/// }
/// ```
pub trait HybridWorkload: Send + Sync {
    /// The result type produced by the workload.
    type Result;

    /// Returns the size of the workload (number of elements to process).
    ///
    /// This is used by the dispatcher to decide between CPU and GPU execution.
    fn workload_size(&self) -> usize;

    /// Executes the workload on CPU.
    ///
    /// This should typically use Rayon or similar for parallel CPU execution.
    fn execute_cpu(&self) -> Self::Result;

    /// Executes the workload on GPU.
    ///
    /// Returns an error if GPU execution fails.
    fn execute_gpu(&self) -> HybridResult<Self::Result>;

    /// Returns the name of the workload (for logging/metrics).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Returns whether GPU execution is supported.
    ///
    /// Override to return `false` if this workload doesn't have a GPU implementation.
    fn supports_gpu(&self) -> bool {
        true
    }

    /// Returns an estimate of memory bytes required for this workload.
    ///
    /// Used by the resource guard to prevent OOM situations.
    fn memory_estimate(&self) -> usize {
        0
    }
}

/// A wrapper to execute any `FnOnce` as a hybrid workload.
pub struct FnWorkload<F, R>
where
    F: FnOnce() -> R + Send + Sync,
{
    cpu_fn: Option<F>,
    size: usize,
    _marker: std::marker::PhantomData<R>,
}

impl<F, R> FnWorkload<F, R>
where
    F: FnOnce() -> R + Send + Sync,
{
    /// Creates a CPU-only workload from a function.
    pub fn cpu_only(f: F, size: usize) -> Self {
        Self {
            cpu_fn: Some(f),
            size,
            _marker: std::marker::PhantomData,
        }
    }
}

/// A boxed hybrid workload for dynamic dispatch.
pub type BoxedWorkload<R> = Box<dyn HybridWorkload<Result = R>>;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestWorkload {
        data: Vec<f32>,
    }

    impl HybridWorkload for TestWorkload {
        type Result = f32;

        fn workload_size(&self) -> usize {
            self.data.len()
        }

        fn execute_cpu(&self) -> Self::Result {
            self.data.iter().sum()
        }

        fn execute_gpu(&self) -> HybridResult<Self::Result> {
            // Simulate GPU execution
            Ok(self.data.iter().sum())
        }

        fn name(&self) -> &str {
            "TestWorkload"
        }
    }

    #[test]
    fn test_workload_cpu() {
        let workload = TestWorkload {
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        assert_eq!(workload.workload_size(), 4);
        assert!((workload.execute_cpu() - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_workload_gpu() {
        let workload = TestWorkload {
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        let result = workload.execute_gpu().unwrap();
        assert!((result - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_workload_name() {
        let workload = TestWorkload { data: vec![] };
        assert_eq!(workload.name(), "TestWorkload");
    }
}
