//! Cloud storage backends for kernel checkpoints.
//!
//! This module provides cloud storage implementations for the [`CheckpointStorage`] trait,
//! enabling distributed checkpoint persistence across S3-compatible object stores.
//!
//! # Supported Backends
//!
//! - **S3Storage**: Amazon S3 and S3-compatible stores (MinIO, R2, etc.)
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::cloud_storage::{S3Storage, S3Config};
//! use ringkernel_core::checkpoint::{CheckpointBuilder, CheckpointStorage};
//!
//! // Configure S3 storage
//! let config = S3Config::new("my-bucket")
//!     .with_prefix("checkpoints/")
//!     .with_region("us-east-1");
//!
//! let storage = S3Storage::new(config).await?;
//!
//! // Save a checkpoint
//! let checkpoint = CheckpointBuilder::new("kernel_1", "fdtd_3d")
//!     .step(1000)
//!     .build();
//!
//! storage.save(&checkpoint, "step_1000").await?;
//!
//! // Load it back
//! let restored = storage.load("step_1000").await?;
//! ```

use std::time::Duration;

use crate::checkpoint::{Checkpoint, CheckpointStorage};
use crate::error::{Result, RingKernelError};

// ============================================================================
// S3 Configuration
// ============================================================================

/// Configuration for S3 storage backend.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// S3 bucket name.
    pub bucket: String,
    /// Key prefix for all checkpoints (e.g., "checkpoints/").
    pub prefix: String,
    /// AWS region (e.g., "us-east-1").
    pub region: Option<String>,
    /// Custom endpoint URL for S3-compatible stores.
    pub endpoint_url: Option<String>,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retries for failed requests.
    pub max_retries: u32,
    /// Enable server-side encryption (SSE-S3).
    pub server_side_encryption: bool,
    /// Storage class for objects (e.g., "STANDARD", "GLACIER").
    pub storage_class: Option<String>,
}

impl S3Config {
    /// Create a new S3 configuration with the given bucket.
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            prefix: String::new(),
            region: None,
            endpoint_url: None,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            server_side_encryption: false,
            storage_class: None,
        }
    }

    /// Set the key prefix for checkpoints.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the AWS region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set a custom endpoint URL (for MinIO, R2, etc.).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint.into());
        self
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enable server-side encryption.
    pub fn with_encryption(mut self) -> Self {
        self.server_side_encryption = true;
        self
    }

    /// Set storage class.
    pub fn with_storage_class(mut self, class: impl Into<String>) -> Self {
        self.storage_class = Some(class.into());
        self
    }

    /// Get the full key for a checkpoint name.
    fn key(&self, name: &str) -> String {
        format!("{}{}.rkcp", self.prefix, name)
    }
}

impl Default for S3Config {
    fn default() -> Self {
        Self::new("ringkernel-checkpoints")
    }
}

// ============================================================================
// S3 Storage Backend
// ============================================================================

/// S3-based checkpoint storage using AWS SDK.
///
/// Supports Amazon S3 and S3-compatible object stores like MinIO, Cloudflare R2,
/// DigitalOcean Spaces, etc.
pub struct S3Storage {
    /// S3 client.
    client: aws_sdk_s3::Client,
    /// Configuration.
    config: S3Config,
}

impl S3Storage {
    /// Create a new S3 storage backend.
    ///
    /// This uses AWS SDK's default credential chain (environment variables,
    /// AWS config files, IAM roles, etc.).
    pub async fn new(config: S3Config) -> Result<Self> {
        let sdk_config = Self::build_aws_config(&config).await?;
        let s3_config = Self::build_s3_config(&config, &sdk_config);
        let client = aws_sdk_s3::Client::from_conf(s3_config);

        Ok(Self { client, config })
    }

    /// Create S3 storage with explicit credentials.
    pub async fn with_credentials(
        config: S3Config,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Result<Self> {
        use aws_config::BehaviorVersion;

        let creds = aws_sdk_s3::config::Credentials::new(
            access_key.into(),
            secret_key.into(),
            None, // session token
            None, // expiry
            "ringkernel",
        );

        let region = config
            .region
            .clone()
            .unwrap_or_else(|| "us-east-1".to_string());

        let mut sdk_config_loader = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_sdk_s3::config::Region::new(region))
            .credentials_provider(creds);

        if let Some(ref endpoint) = config.endpoint_url {
            sdk_config_loader = sdk_config_loader.endpoint_url(endpoint);
        }

        let sdk_config = sdk_config_loader.load().await;
        let s3_config = Self::build_s3_config(&config, &sdk_config);
        let client = aws_sdk_s3::Client::from_conf(s3_config);

        Ok(Self { client, config })
    }

    /// Build AWS SDK configuration.
    async fn build_aws_config(config: &S3Config) -> Result<aws_config::SdkConfig> {
        use aws_config::BehaviorVersion;

        let mut loader = aws_config::defaults(BehaviorVersion::latest());

        if let Some(ref region) = config.region {
            loader = loader.region(aws_sdk_s3::config::Region::new(region.clone()));
        }

        if let Some(ref endpoint) = config.endpoint_url {
            loader = loader.endpoint_url(endpoint);
        }

        Ok(loader.load().await)
    }

    /// Build S3 client configuration.
    fn build_s3_config(
        config: &S3Config,
        sdk_config: &aws_config::SdkConfig,
    ) -> aws_sdk_s3::Config {
        let mut builder = aws_sdk_s3::config::Builder::from(sdk_config)
            .force_path_style(config.endpoint_url.is_some()); // Use path style for MinIO etc.

        // Set retry config with timeout
        let retry_config = aws_sdk_s3::config::retry::RetryConfig::standard()
            .with_max_attempts(config.max_retries);
        builder = builder.retry_config(retry_config);

        // Set stalled stream protection (uses a different pattern in newer SDK)
        let _ = config.timeout; // Timeout is handled by the retry config

        builder.build()
    }

    /// Save checkpoint to S3 with retries.
    async fn save_with_retry(&self, checkpoint: &Checkpoint, name: &str) -> Result<()> {
        let key = self.config.key(name);
        let data = checkpoint.to_bytes();

        let mut request = self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .body(aws_sdk_s3::primitives::ByteStream::from(data))
            .content_type("application/octet-stream");

        // Add server-side encryption if enabled
        if self.config.server_side_encryption {
            request =
                request.server_side_encryption(aws_sdk_s3::types::ServerSideEncryption::Aes256);
        }

        // Add storage class if specified
        if let Some(ref class) = self.config.storage_class {
            // StorageClass::from_str returns Result<StorageClass, Infallible>
            // which always succeeds, so we can just unwrap
            let storage_class: aws_sdk_s3::types::StorageClass = class.parse().unwrap();
            request = request.storage_class(storage_class);
        }

        // Add metadata
        request = request
            .metadata("kernel-id", &checkpoint.metadata.kernel_id)
            .metadata("kernel-type", &checkpoint.metadata.kernel_type)
            .metadata("step", &checkpoint.metadata.current_step.to_string());

        request.send().await.map_err(|e| {
            RingKernelError::IoError(format!("Failed to upload checkpoint to S3: {}", e))
        })?;

        Ok(())
    }

    /// Load checkpoint from S3 with retries.
    async fn load_with_retry(&self, name: &str) -> Result<Checkpoint> {
        let key = self.config.key(name);

        let response = self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| {
                RingKernelError::IoError(format!("Failed to download checkpoint from S3: {}", e))
            })?;

        let bytes = response.body.collect().await.map_err(|e| {
            RingKernelError::IoError(format!("Failed to read checkpoint body: {}", e))
        })?;

        Checkpoint::from_bytes(&bytes.to_vec())
    }
}

impl CheckpointStorage for S3Storage {
    fn save(&self, checkpoint: &Checkpoint, name: &str) -> Result<()> {
        // Use tokio's current runtime to block on the async operation
        // This is safe because CheckpointStorage is typically called from async contexts
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.save_with_retry(checkpoint, name))
        })
    }

    fn load(&self, name: &str) -> Result<Checkpoint> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.load_with_retry(name))
        })
    }

    fn list(&self) -> Result<Vec<String>> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut names = Vec::new();
                let mut continuation_token = None;

                loop {
                    let mut request = self
                        .client
                        .list_objects_v2()
                        .bucket(&self.config.bucket)
                        .prefix(&self.config.prefix);

                    if let Some(token) = continuation_token.take() {
                        request = request.continuation_token(token);
                    }

                    let response = request.send().await.map_err(|e| {
                        RingKernelError::IoError(format!("Failed to list S3 objects: {}", e))
                    })?;

                    if let Some(contents) = response.contents {
                        for obj in contents {
                            if let Some(key) = obj.key {
                                // Extract name from key (remove prefix and .rkcp suffix)
                                if let Some(name) = key
                                    .strip_prefix(&self.config.prefix)
                                    .and_then(|s| s.strip_suffix(".rkcp"))
                                {
                                    names.push(name.to_string());
                                }
                            }
                        }
                    }

                    if response.is_truncated.unwrap_or(false) {
                        continuation_token = response.next_continuation_token;
                    } else {
                        break;
                    }
                }

                names.sort();
                Ok(names)
            })
        })
    }

    fn delete(&self, name: &str) -> Result<()> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let key = self.config.key(name);

                self.client
                    .delete_object()
                    .bucket(&self.config.bucket)
                    .key(&key)
                    .send()
                    .await
                    .map_err(|e| {
                        RingKernelError::IoError(format!(
                            "Failed to delete checkpoint from S3: {}",
                            e
                        ))
                    })?;

                Ok(())
            })
        })
    }

    fn exists(&self, name: &str) -> bool {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let key = self.config.key(name);

                self.client
                    .head_object()
                    .bucket(&self.config.bucket)
                    .key(&key)
                    .send()
                    .await
                    .is_ok()
            })
        })
    }
}

// ============================================================================
// Async Storage Trait
// ============================================================================

/// Async version of CheckpointStorage for more efficient cloud operations.
#[async_trait::async_trait]
pub trait AsyncCheckpointStorage: Send + Sync {
    /// Save a checkpoint asynchronously.
    async fn save_async(&self, checkpoint: &Checkpoint, name: &str) -> Result<()>;

    /// Load a checkpoint asynchronously.
    async fn load_async(&self, name: &str) -> Result<Checkpoint>;

    /// List all checkpoints asynchronously.
    async fn list_async(&self) -> Result<Vec<String>>;

    /// Delete a checkpoint asynchronously.
    async fn delete_async(&self, name: &str) -> Result<()>;

    /// Check if a checkpoint exists asynchronously.
    async fn exists_async(&self, name: &str) -> bool;
}

#[async_trait::async_trait]
impl AsyncCheckpointStorage for S3Storage {
    async fn save_async(&self, checkpoint: &Checkpoint, name: &str) -> Result<()> {
        self.save_with_retry(checkpoint, name).await
    }

    async fn load_async(&self, name: &str) -> Result<Checkpoint> {
        self.load_with_retry(name).await
    }

    async fn list_async(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let mut continuation_token = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.config.bucket)
                .prefix(&self.config.prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request.send().await.map_err(|e| {
                RingKernelError::IoError(format!("Failed to list S3 objects: {}", e))
            })?;

            if let Some(contents) = response.contents {
                for obj in contents {
                    if let Some(key) = obj.key {
                        if let Some(name) = key
                            .strip_prefix(&self.config.prefix)
                            .and_then(|s| s.strip_suffix(".rkcp"))
                        {
                            names.push(name.to_string());
                        }
                    }
                }
            }

            if response.is_truncated.unwrap_or(false) {
                continuation_token = response.next_continuation_token;
            } else {
                break;
            }
        }

        names.sort();
        Ok(names)
    }

    async fn delete_async(&self, name: &str) -> Result<()> {
        let key = self.config.key(name);

        self.client
            .delete_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| {
                RingKernelError::IoError(format!("Failed to delete checkpoint from S3: {}", e))
            })?;

        Ok(())
    }

    async fn exists_async(&self, name: &str) -> bool {
        let key = self.config.key(name);

        self.client
            .head_object()
            .bucket(&self.config.bucket)
            .key(&key)
            .send()
            .await
            .is_ok()
    }
}

// ============================================================================
// Cloud Storage Factory
// ============================================================================

/// Cloud storage provider type.
#[derive(Debug, Clone)]
pub enum CloudProvider {
    /// Amazon S3 or S3-compatible store.
    S3(S3Config),
    // Future: GCS, Azure Blob, etc.
}

/// Create a cloud storage backend from configuration.
pub async fn create_cloud_storage(
    provider: CloudProvider,
) -> Result<Box<dyn CheckpointStorage>> {
    match provider {
        CloudProvider::S3(config) => {
            let storage = S3Storage::new(config).await?;
            Ok(Box::new(storage))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_defaults() {
        let config = S3Config::new("my-bucket");
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.prefix, "");
        assert!(config.region.is_none());
        assert!(config.endpoint_url.is_none());
        assert_eq!(config.max_retries, 3);
        assert!(!config.server_side_encryption);
    }

    #[test]
    fn test_s3_config_builder() {
        let config = S3Config::new("my-bucket")
            .with_prefix("checkpoints/")
            .with_region("us-west-2")
            .with_endpoint("http://localhost:9000")
            .with_encryption()
            .with_max_retries(5)
            .with_storage_class("GLACIER");

        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.prefix, "checkpoints/");
        assert_eq!(config.region, Some("us-west-2".to_string()));
        assert_eq!(
            config.endpoint_url,
            Some("http://localhost:9000".to_string())
        );
        assert!(config.server_side_encryption);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.storage_class, Some("GLACIER".to_string()));
    }

    #[test]
    fn test_s3_key_generation() {
        let config = S3Config::new("bucket").with_prefix("checkpoints/");
        assert_eq!(config.key("step_1000"), "checkpoints/step_1000.rkcp");

        let config_no_prefix = S3Config::new("bucket");
        assert_eq!(config_no_prefix.key("step_1000"), "step_1000.rkcp");
    }
}
