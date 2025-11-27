//! Shader compilation and management.

use std::sync::Arc;

use ringkernel_core::error::{Result, RingKernelError};

use crate::adapter::WgpuAdapter;

/// Compiled compute pipeline.
pub struct ComputePipeline {
    /// The wgpu pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pipeline layout.
    pipeline_layout: wgpu::PipelineLayout,
    /// Workgroup size.
    workgroup_size: (u32, u32, u32),
}

impl ComputePipeline {
    /// Create a new compute pipeline from WGSL source.
    pub fn new(
        adapter: &WgpuAdapter,
        wgsl_source: &str,
        entry_point: &str,
        workgroup_size: (u32, u32, u32),
    ) -> Result<Self> {
        let device = adapter.device();

        // Create shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RingKernel Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // Create bind group layout for control block and queues
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RingKernel Bind Group Layout"),
            entries: &[
                // Control block
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input queue
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output queue
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RingKernel Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RingKernel Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            pipeline_layout,
            workgroup_size,
        })
    }

    /// Get the pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Get bind group layout.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get workgroup size.
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        self.workgroup_size
    }
}

/// Create a bind group for the kernel.
pub fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    control_block: wgpu::BindingResource,
    input_queue: wgpu::BindingResource,
    output_queue: wgpu::BindingResource,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("RingKernel Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: control_block,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_queue,
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_queue,
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RING_KERNEL_WGSL_TEMPLATE;

    #[tokio::test]
    #[ignore] // May not have GPU in CI
    async fn test_pipeline_creation() {
        let adapter = WgpuAdapter::new().await.unwrap();
        let pipeline = ComputePipeline::new(
            &adapter,
            RING_KERNEL_WGSL_TEMPLATE,
            "main",
            (256, 1, 1),
        )
        .unwrap();

        assert_eq!(pipeline.workgroup_size(), (256, 1, 1));
    }
}
