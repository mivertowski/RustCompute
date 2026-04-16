//! LLM Provider Bridge — FR-012
//!
//! Integration layer for calling LLMs from within the GPU actor system.
//! Each provider implements the `LlmProvider` trait. Requests and responses
//! flow as K2K messages, making LLM calls a natural actor interaction.
//!
//! # Supported Providers
//!
//! - **OpenAI**: GPT-4, GPT-3.5, embeddings
//! - **Anthropic**: Claude (messages API, tool use)
//! - **Local**: Ollama, vLLM, llama.cpp via HTTP
//!
//! # Architecture
//!
//! ```text
//! GPU Actor → K2K "LLM Request" → LLM Bridge Actor → HTTP → Provider
//!                                                          ↓
//! GPU Actor ← K2K "LLM Response" ← LLM Bridge Actor ← HTTP ← Provider
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// An LLM completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus", "llama-3-70b").
    pub model: String,
    /// Messages in the conversation.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Temperature (0.0 = deterministic, 1.0 = creative).
    pub temperature: Option<f32>,
    /// System prompt.
    pub system: Option<String>,
    /// Tool/function definitions (for function calling).
    pub tools: Vec<ToolDefinition>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Request metadata (for tracking, routing).
    pub metadata: HashMap<String, String>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: "gpt-4".to_string(),
            messages: Vec::new(),
            max_tokens: Some(1024),
            temperature: Some(0.7),
            system: None,
            tools: Vec::new(),
            stop: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl CompletionRequest {
    /// Create a simple completion request.
    pub fn new(model: impl Into<String>, user_message: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: vec![ChatMessage::user(user_message)],
            ..Default::default()
        }
    }

    /// Add a system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Add a tool definition.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }
}

/// A chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant", "tool".
    pub role: String,
    /// Message content.
    pub content: String,
    /// Tool call ID (for tool responses).
    pub tool_call_id: Option<String>,
    /// Tool calls made by the assistant.
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    /// Create a tool response message.
    pub fn tool_response(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_call_id: Some(call_id.into()),
            tool_calls: Vec::new(),
        }
    }
}

/// A tool/function definition for LLM function calling.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Parameter schema (JSON Schema as string).
    pub parameters_schema: String,
}

/// A tool call made by the LLM.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Unique call ID.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments (JSON string).
    pub arguments: String,
}

/// An LLM completion response.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Generated text content.
    pub content: String,
    /// Model used.
    pub model: String,
    /// Tool calls (if function calling was used).
    pub tool_calls: Vec<ToolCall>,
    /// Finish reason.
    pub finish_reason: FinishReason,
    /// Token usage.
    pub usage: TokenUsage,
    /// Response latency.
    pub latency: Duration,
    /// Provider-specific response ID.
    pub response_id: Option<String>,
}

/// Why the LLM stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Reached natural end.
    Stop,
    /// Hit max_tokens limit.
    MaxTokens,
    /// Used a tool/function call.
    ToolUse,
    /// Content was filtered.
    ContentFilter,
    /// Unknown reason.
    Unknown,
}

/// Token usage tracking.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    /// Prompt/input tokens.
    pub prompt_tokens: u32,
    /// Completion/output tokens.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

impl TokenUsage {
    /// Estimated cost in USD (rough, provider-dependent).
    pub fn estimated_cost(&self, prompt_cost_per_1k: f64, completion_cost_per_1k: f64) -> f64 {
        (self.prompt_tokens as f64 / 1000.0 * prompt_cost_per_1k)
            + (self.completion_tokens as f64 / 1000.0 * completion_cost_per_1k)
    }
}

/// An embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Model for embeddings (e.g., "text-embedding-3-small").
    pub model: String,
    /// Texts to embed.
    pub inputs: Vec<String>,
    /// Dimensionality (if model supports it).
    pub dimensions: Option<u32>,
}

/// An embedding response.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// Embedding vectors (one per input).
    pub embeddings: Vec<Vec<f32>>,
    /// Token usage.
    pub usage: TokenUsage,
    /// Response latency.
    pub latency: Duration,
}

/// Trait for LLM provider implementations.
pub trait LlmProvider: Send + Sync {
    /// Provider name (e.g., "openai", "anthropic", "ollama").
    fn name(&self) -> &str;

    /// Send a completion request.
    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, LlmError>;

    /// Generate embeddings.
    fn embed(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError>;

    /// Check if the provider is available/connected.
    fn is_available(&self) -> bool;

    /// Get provider metrics.
    fn metrics(&self) -> LlmProviderMetrics;
}

/// LLM provider metrics.
#[derive(Debug, Clone, Default)]
pub struct LlmProviderMetrics {
    /// Total requests made.
    pub total_requests: u64,
    /// Total errors.
    pub total_errors: u64,
    /// Total tokens consumed.
    pub total_tokens: u64,
    /// Total estimated cost (USD).
    pub total_cost_usd: f64,
    /// Average latency.
    pub avg_latency: Duration,
    /// Rate limit remaining.
    pub rate_limit_remaining: Option<u32>,
}

/// LLM errors.
#[derive(Debug, Clone)]
pub enum LlmError {
    /// Provider not available.
    Unavailable(String),
    /// Rate limited.
    RateLimited { retry_after: Duration },
    /// Authentication failed.
    AuthError(String),
    /// Request invalid.
    InvalidRequest(String),
    /// Model not found.
    ModelNotFound(String),
    /// Content filtered.
    ContentFiltered,
    /// Timeout.
    Timeout(Duration),
    /// Provider error.
    ProviderError(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(msg) => write!(f, "Provider unavailable: {}", msg),
            Self::RateLimited { retry_after } => write!(f, "Rate limited, retry after {:?}", retry_after),
            Self::AuthError(msg) => write!(f, "Auth error: {}", msg),
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::ModelNotFound(model) => write!(f, "Model not found: {}", model),
            Self::ContentFiltered => write!(f, "Content filtered"),
            Self::Timeout(d) => write!(f, "Timeout after {:?}", d),
            Self::ProviderError(msg) => write!(f, "Provider error: {}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

// ============================================================================
// Echo Provider (for testing)
// ============================================================================

/// Test LLM provider that echoes the input. Useful for testing the bridge.
pub struct EchoProvider {
    total_requests: std::sync::atomic::AtomicU64,
}

impl EchoProvider {
    /// Create a new echo provider.
    pub fn new() -> Self {
        Self {
            total_requests: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Default for EchoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for EchoProvider {
    fn name(&self) -> &str {
        "echo"
    }

    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LlmError> {
        self.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let content = request
            .messages
            .last()
            .map(|m| format!("Echo: {}", m.content))
            .unwrap_or_else(|| "Echo: (empty)".to_string());

        Ok(CompletionResponse {
            content,
            model: request.model.clone(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 10,
                total_tokens: 20,
            },
            latency: Duration::from_micros(100),
            response_id: Some("echo-001".to_string()),
        })
    }

    fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let dim = request.dimensions.unwrap_or(384) as usize;
        let embeddings: Vec<Vec<f32>> = request
            .inputs
            .iter()
            .map(|input| {
                // Deterministic pseudo-embedding based on input hash
                let hash = input.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                (0..dim)
                    .map(|i| ((hash.wrapping_add(i as u64) % 1000) as f32 / 1000.0) - 0.5)
                    .collect()
            })
            .collect();

        Ok(EmbeddingResponse {
            embeddings,
            usage: TokenUsage {
                prompt_tokens: request.inputs.len() as u32 * 5,
                completion_tokens: 0,
                total_tokens: request.inputs.len() as u32 * 5,
            },
            latency: Duration::from_micros(50),
        })
    }

    fn is_available(&self) -> bool {
        true
    }

    fn metrics(&self) -> LlmProviderMetrics {
        LlmProviderMetrics {
            total_requests: self.total_requests.load(std::sync::atomic::Ordering::Relaxed),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_provider_complete() {
        let provider = EchoProvider::new();
        let request = CompletionRequest::new("test-model", "Hello, world!");
        let response = provider.complete(&request).unwrap();

        assert!(response.content.contains("Hello, world!"));
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.total_tokens, 20);
    }

    #[test]
    fn test_echo_provider_embed() {
        let provider = EchoProvider::new();
        let request = EmbeddingRequest {
            model: "test-embed".to_string(),
            inputs: vec!["hello".to_string(), "world".to_string()],
            dimensions: Some(128),
        };
        let response = provider.embed(&request).unwrap();

        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0].len(), 128);
    }

    #[test]
    fn test_completion_request_builder() {
        let req = CompletionRequest::new("gpt-4", "Analyze this audit finding")
            .with_system("You are an ISA compliance expert")
            .with_max_tokens(2048)
            .with_temperature(0.0);

        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.system.unwrap(), "You are an ISA compliance expert");
        assert_eq!(req.max_tokens, Some(2048));
        assert_eq!(req.temperature, Some(0.0));
    }

    #[test]
    fn test_chat_message_types() {
        let user = ChatMessage::user("question");
        assert_eq!(user.role, "user");

        let assistant = ChatMessage::assistant("answer");
        assert_eq!(assistant.role, "assistant");

        let tool = ChatMessage::tool_response("call-123", "result");
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.unwrap(), "call-123");
    }

    #[test]
    fn test_token_cost_estimation() {
        let usage = TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
        };
        // GPT-4 pricing: ~$0.03/1K prompt, ~$0.06/1K completion
        let cost = usage.estimated_cost(0.03, 0.06);
        assert!((cost - 0.06).abs() < 0.001); // $0.03 + $0.03 = $0.06
    }

    #[test]
    fn test_provider_metrics() {
        let provider = EchoProvider::new();

        provider.complete(&CompletionRequest::new("m", "a")).unwrap();
        provider.complete(&CompletionRequest::new("m", "b")).unwrap();

        let m = provider.metrics();
        assert_eq!(m.total_requests, 2);
    }

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition {
            name: "graph_query".to_string(),
            description: "Query the knowledge graph".to_string(),
            parameters_schema: r#"{"type":"object","properties":{"query":{"type":"string"}}}"#.to_string(),
        };
        assert_eq!(tool.name, "graph_query");
    }
}
