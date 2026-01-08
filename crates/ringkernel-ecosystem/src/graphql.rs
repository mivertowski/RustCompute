//! GraphQL integration with subscription support for persistent GPU kernels.
//!
//! This module provides async-graphql integration for RingKernel, enabling
//! real-time GPU kernel communication via GraphQL subscriptions.
//!
//! # Features
//!
//! - Query kernel status and statistics
//! - Mutations for sending commands to kernels
//! - Subscriptions for real-time kernel events and progress updates
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use ringkernel_ecosystem::graphql::{
//!     create_schema, GraphQLState, KernelSchema
//! };
//!
//! // Create schema with your persistent handle
//! let state = GraphQLState::new(Arc::new(handle));
//! let schema = create_schema(state);
//!
//! // Use with axum
//! let app = graphql_router(state);
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_graphql::{
    Context, Enum, InputObject, Object, Result as GqlResult, Schema, SimpleObject, Subscription, ID,
};
use tokio::sync::broadcast;

use crate::persistent::{
    CommandId, PersistentCommand, PersistentConfig, PersistentHandle, PersistentResponse,
    PersistentStats,
};

// ============================================================================
// GraphQL Types
// ============================================================================

/// Kernel status information.
#[derive(Debug, Clone, SimpleObject)]
pub struct KernelStatus {
    /// Kernel identifier.
    pub id: String,
    /// Whether the kernel is currently running.
    pub running: bool,
    /// Current simulation step.
    pub current_step: u64,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Pending commands in queue.
    pub pending_commands: u32,
    /// Average command latency in microseconds.
    pub avg_latency_us: f64,
}

/// Kernel statistics snapshot.
#[derive(Debug, Clone, SimpleObject)]
pub struct KernelStatsResponse {
    /// Current step in the simulation.
    pub current_step: u64,
    /// Steps remaining in current batch.
    pub steps_remaining: u64,
    /// Total messages processed since start.
    pub messages_processed: u64,
    /// Total energy in the system (for physics simulations).
    pub total_energy: f32,
    /// K2K messages sent.
    pub k2k_sent: u64,
    /// K2K messages received.
    pub k2k_received: u64,
    /// Whether the kernel is running.
    pub is_running: bool,
    /// Pending commands.
    pub pending_commands: u32,
}

impl From<PersistentStats> for KernelStatsResponse {
    fn from(stats: PersistentStats) -> Self {
        Self {
            current_step: stats.current_step,
            steps_remaining: stats.steps_remaining,
            messages_processed: stats.messages_processed,
            total_energy: stats.total_energy,
            k2k_sent: stats.k2k_sent,
            k2k_received: stats.k2k_received,
            is_running: stats.is_running,
            pending_commands: stats.pending_commands,
        }
    }
}

/// Command acknowledgment response.
#[derive(Debug, Clone, SimpleObject)]
pub struct CommandAck {
    /// Command ID that was acknowledged.
    pub command_id: u64,
    /// Whether the command was successful.
    pub success: bool,
    /// Optional message.
    pub message: Option<String>,
}

/// Progress update from a running command.
#[derive(Debug, Clone, SimpleObject)]
pub struct ProgressUpdate {
    /// Command ID being tracked.
    pub command_id: u64,
    /// Steps completed.
    pub completed: u64,
    /// Total steps remaining.
    pub remaining: u64,
    /// Estimated completion percentage.
    pub percentage: f32,
}

/// Kernel event types for subscriptions.
#[derive(Debug, Clone, Enum, Copy, Eq, PartialEq)]
pub enum KernelEventType {
    /// Command was acknowledged.
    Ack,
    /// Progress update on a command.
    Progress,
    /// Statistics update.
    Stats,
    /// An error occurred.
    Error,
    /// Kernel was terminated.
    Terminated,
    /// Custom event.
    Custom,
}

/// Kernel event for subscriptions.
#[derive(Debug, Clone, SimpleObject)]
pub struct KernelEvent {
    /// Event type.
    pub event_type: KernelEventType,
    /// Associated command ID, if any.
    pub command_id: Option<u64>,
    /// Current step when event occurred.
    pub current_step: u64,
    /// Event message or details.
    pub message: Option<String>,
    /// Progress information, if applicable.
    pub progress: Option<ProgressUpdate>,
}

impl From<PersistentResponse> for KernelEvent {
    fn from(response: PersistentResponse) -> Self {
        match response {
            PersistentResponse::Ack { cmd_id } => Self {
                event_type: KernelEventType::Ack,
                command_id: Some(cmd_id.as_u64()),
                current_step: 0,
                message: Some("Command acknowledged".to_string()),
                progress: None,
            },
            PersistentResponse::Progress {
                cmd_id,
                current_step,
                remaining,
            } => {
                let completed = current_step;
                let total = current_step + remaining;
                let percentage = if total > 0 {
                    (completed as f32 / total as f32) * 100.0
                } else {
                    0.0
                };
                Self {
                    event_type: KernelEventType::Progress,
                    command_id: Some(cmd_id.as_u64()),
                    current_step,
                    message: None,
                    progress: Some(ProgressUpdate {
                        command_id: cmd_id.as_u64(),
                        completed,
                        remaining,
                        percentage,
                    }),
                }
            }
            PersistentResponse::Stats { cmd_id, stats } => Self {
                event_type: KernelEventType::Stats,
                command_id: Some(cmd_id.as_u64()),
                current_step: stats.current_step,
                message: Some(format!(
                    "Stats at step {} ({} messages processed)",
                    stats.current_step, stats.messages_processed
                )),
                progress: None,
            },
            PersistentResponse::Error {
                cmd_id,
                code,
                message,
            } => Self {
                event_type: KernelEventType::Error,
                command_id: Some(cmd_id.as_u64()),
                current_step: 0,
                message: Some(format!("Error {}: {}", code, message)),
                progress: None,
            },
            PersistentResponse::Terminated { final_step } => Self {
                event_type: KernelEventType::Terminated,
                command_id: None,
                current_step: final_step,
                message: Some(format!("Kernel terminated after {} steps", final_step)),
                progress: None,
            },
            PersistentResponse::Custom {
                cmd_id, type_id, ..
            } => Self {
                event_type: KernelEventType::Custom,
                command_id: Some(cmd_id.as_u64()),
                current_step: 0,
                message: Some(format!("Custom event type {}", type_id)),
                progress: None,
            },
        }
    }
}

// ============================================================================
// Input Types
// ============================================================================

/// Input for running simulation steps.
#[derive(Debug, Clone, InputObject)]
pub struct RunStepsInput {
    /// Number of steps to run.
    pub count: u64,
}

/// Input for injecting an impulse/source.
#[derive(Debug, Clone, InputObject)]
pub struct InjectInput {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
    /// Z coordinate (for 3D simulations).
    #[graphql(default = 0)]
    pub z: u32,
    /// Amplitude of the impulse.
    pub amplitude: f32,
}

// ============================================================================
// GraphQL State
// ============================================================================

/// Type-erased handle wrapper for GraphQL operations.
pub type DynPersistentHandle = Arc<dyn PersistentHandle>;

/// Shared state for GraphQL operations.
#[derive(Clone)]
pub struct GraphQLState {
    /// Handle to the persistent kernel (type-erased).
    handle: DynPersistentHandle,
    /// Broadcast channel for kernel events.
    event_tx: broadcast::Sender<KernelEvent>,
}

impl GraphQLState {
    /// Create new GraphQL state from a persistent handle.
    pub fn new<H: PersistentHandle + 'static>(handle: Arc<H>) -> Self {
        let (event_tx, _) = broadcast::channel(1024);
        Self {
            handle: handle as DynPersistentHandle,
            event_tx,
        }
    }

    /// Subscribe to kernel events.
    pub fn subscribe(&self) -> broadcast::Receiver<KernelEvent> {
        self.event_tx.subscribe()
    }

    /// Publish a kernel event.
    pub fn publish(&self, event: KernelEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Get the handle.
    pub fn handle(&self) -> &DynPersistentHandle {
        &self.handle
    }
}

// ============================================================================
// Query Root
// ============================================================================

/// GraphQL Query root for kernel operations.
pub struct KernelQuery;

#[Object]
impl KernelQuery {
    /// Get the current kernel status.
    async fn status(&self, ctx: &Context<'_>) -> GqlResult<KernelStatus> {
        let state = ctx.data::<GraphQLState>()?;
        let stats = state.handle.stats();

        Ok(KernelStatus {
            id: state.handle.kernel_id().to_string(),
            running: state.handle.is_running(),
            current_step: stats.current_step,
            messages_processed: stats.messages_processed,
            pending_commands: stats.pending_commands,
            avg_latency_us: 0.0, // Would need tracking infrastructure
        })
    }

    /// Get detailed kernel statistics.
    async fn stats(&self, ctx: &Context<'_>) -> GqlResult<KernelStatsResponse> {
        let state = ctx.data::<GraphQLState>()?;
        let stats = state.handle.stats();
        Ok(stats.into())
    }

    /// Check if the kernel is running.
    async fn is_running(&self, ctx: &Context<'_>) -> GqlResult<bool> {
        let state = ctx.data::<GraphQLState>()?;
        Ok(state.handle.is_running())
    }

    /// Get the kernel ID.
    async fn kernel_id(&self, ctx: &Context<'_>) -> GqlResult<ID> {
        let state = ctx.data::<GraphQLState>()?;
        Ok(ID(state.handle.kernel_id().to_string()))
    }

    /// Get the kernel configuration.
    async fn config(&self, ctx: &Context<'_>) -> GqlResult<KernelConfigResponse> {
        let state = ctx.data::<GraphQLState>()?;
        let config = state.handle.config();
        Ok(KernelConfigResponse::from(config))
    }
}

/// Kernel configuration response.
#[derive(Debug, Clone, SimpleObject)]
pub struct KernelConfigResponse {
    /// H2K queue capacity.
    pub h2k_capacity: i32,
    /// K2H queue capacity.
    pub k2h_capacity: i32,
    /// Progress interval in steps.
    pub progress_interval: i32,
    /// Poll interval in microseconds.
    pub poll_interval_us: i32,
    /// Command timeout in seconds.
    pub command_timeout_secs: i32,
    /// Max in-flight commands.
    pub max_in_flight: i32,
}

impl From<&PersistentConfig> for KernelConfigResponse {
    fn from(config: &PersistentConfig) -> Self {
        Self {
            h2k_capacity: config.h2k_capacity as i32,
            k2h_capacity: config.k2h_capacity as i32,
            progress_interval: config.progress_interval as i32,
            poll_interval_us: config.poll_interval.as_micros() as i32,
            command_timeout_secs: config.command_timeout.as_secs() as i32,
            max_in_flight: config.max_in_flight as i32,
        }
    }
}

// ============================================================================
// Mutation Root
// ============================================================================

/// GraphQL Mutation root for kernel commands.
pub struct KernelMutation;

#[Object]
impl KernelMutation {
    /// Run a specified number of simulation steps.
    async fn run_steps(&self, ctx: &Context<'_>, input: RunStepsInput) -> GqlResult<CommandAck> {
        let state = ctx.data::<GraphQLState>()?;

        if !state.handle.is_running() {
            return Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some("Kernel is not running".to_string()),
            });
        }

        let cmd = PersistentCommand::RunSteps { count: input.count };

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => {
                state.publish(KernelEvent {
                    event_type: KernelEventType::Ack,
                    command_id: Some(cmd_id.as_u64()),
                    current_step: state.handle.stats().current_step,
                    message: Some(format!("Running {} steps", input.count)),
                    progress: None,
                });

                Ok(CommandAck {
                    command_id: cmd_id.as_u64(),
                    success: true,
                    message: Some(format!("Started running {} steps", input.count)),
                })
            }
            Err(e) => Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some(e.to_string()),
            }),
        }
    }

    /// Inject an impulse at specified coordinates.
    async fn inject(&self, ctx: &Context<'_>, input: InjectInput) -> GqlResult<CommandAck> {
        let state = ctx.data::<GraphQLState>()?;

        if !state.handle.is_running() {
            return Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some("Kernel is not running".to_string()),
            });
        }

        let cmd = PersistentCommand::Inject {
            position: (input.x, input.y, input.z),
            value: input.amplitude,
        };

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => Ok(CommandAck {
                command_id: cmd_id.as_u64(),
                success: true,
                message: Some(format!(
                    "Injected amplitude {} at ({}, {}, {})",
                    input.amplitude, input.x, input.y, input.z
                )),
            }),
            Err(e) => Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some(e.to_string()),
            }),
        }
    }

    /// Pause the kernel.
    async fn pause(&self, ctx: &Context<'_>) -> GqlResult<CommandAck> {
        let state = ctx.data::<GraphQLState>()?;
        let cmd = PersistentCommand::Pause;

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => Ok(CommandAck {
                command_id: cmd_id.as_u64(),
                success: true,
                message: Some("Kernel paused".to_string()),
            }),
            Err(e) => Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some(e.to_string()),
            }),
        }
    }

    /// Resume the kernel.
    async fn resume(&self, ctx: &Context<'_>) -> GqlResult<CommandAck> {
        let state = ctx.data::<GraphQLState>()?;
        let cmd = PersistentCommand::Resume;

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => Ok(CommandAck {
                command_id: cmd_id.as_u64(),
                success: true,
                message: Some("Kernel resumed".to_string()),
            }),
            Err(e) => Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some(e.to_string()),
            }),
        }
    }

    /// Request kernel statistics.
    async fn get_stats(&self, ctx: &Context<'_>) -> GqlResult<CommandAck> {
        let state = ctx.data::<GraphQLState>()?;
        let cmd = PersistentCommand::GetStats;

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => Ok(CommandAck {
                command_id: cmd_id.as_u64(),
                success: true,
                message: Some("Stats requested".to_string()),
            }),
            Err(e) => Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some(e.to_string()),
            }),
        }
    }

    /// Terminate the kernel.
    async fn terminate(&self, ctx: &Context<'_>) -> GqlResult<CommandAck> {
        let state = ctx.data::<GraphQLState>()?;
        let cmd = PersistentCommand::Terminate;

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => {
                state.publish(KernelEvent {
                    event_type: KernelEventType::Terminated,
                    command_id: Some(cmd_id.as_u64()),
                    current_step: state.handle.stats().current_step,
                    message: Some("Kernel termination initiated".to_string()),
                    progress: None,
                });

                Ok(CommandAck {
                    command_id: cmd_id.as_u64(),
                    success: true,
                    message: Some("Kernel termination initiated".to_string()),
                })
            }
            Err(e) => Ok(CommandAck {
                command_id: 0,
                success: false,
                message: Some(e.to_string()),
            }),
        }
    }
}

// ============================================================================
// Subscription Root
// ============================================================================

use futures::stream::Stream;

/// GraphQL Subscription root for real-time kernel events.
pub struct KernelSubscription;

#[Subscription]
impl KernelSubscription {
    /// Subscribe to all kernel events.
    async fn events(
        &self,
        ctx: &Context<'_>,
    ) -> async_graphql::Result<impl Stream<Item = KernelEvent> + '_> {
        let state = ctx.data::<GraphQLState>()?;
        let mut rx = state.subscribe();

        let stream = async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => yield event,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        };

        Ok(stream)
    }

    /// Subscribe to progress updates only.
    async fn progress(
        &self,
        ctx: &Context<'_>,
    ) -> async_graphql::Result<impl Stream<Item = ProgressUpdate> + '_> {
        let state = ctx.data::<GraphQLState>()?;
        let mut rx = state.subscribe();

        let stream = async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if let Some(progress) = event.progress {
                            yield progress;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        };

        Ok(stream)
    }

    /// Subscribe to kernel status updates at a specified interval.
    async fn status_updates(
        &self,
        ctx: &Context<'_>,
        #[graphql(default = 1000)] interval_ms: i32,
    ) -> async_graphql::Result<impl Stream<Item = KernelStatus> + '_> {
        let state = ctx.data::<GraphQLState>()?.clone();
        let interval = Duration::from_millis(interval_ms.max(100) as u64);

        let stream = async_stream::stream! {
            loop {
                let stats = state.handle.stats();
                yield KernelStatus {
                    id: state.handle.kernel_id().to_string(),
                    running: state.handle.is_running(),
                    current_step: stats.current_step,
                    messages_processed: stats.messages_processed,
                    pending_commands: stats.pending_commands,
                    avg_latency_us: 0.0,
                };

                tokio::time::sleep(interval).await;

                if !state.handle.is_running() {
                    break;
                }
            }
        };

        Ok(stream)
    }

    /// Subscribe to a specific command's progress.
    async fn command_progress(
        &self,
        ctx: &Context<'_>,
        command_id: u64,
    ) -> async_graphql::Result<impl Stream<Item = KernelEvent> + '_> {
        let state = ctx.data::<GraphQLState>()?;
        let mut rx = state.subscribe();

        let stream = async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if event.command_id == Some(command_id) {
                            let is_terminal = matches!(
                                event.event_type,
                                KernelEventType::Error | KernelEventType::Terminated
                            );

                            yield event;

                            if is_terminal {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        };

        Ok(stream)
    }
}

// ============================================================================
// Schema Creation
// ============================================================================

/// The complete GraphQL schema type.
pub type KernelSchema = Schema<KernelQuery, KernelMutation, KernelSubscription>;

/// Create a GraphQL schema for kernel operations.
pub fn create_schema(state: GraphQLState) -> KernelSchema {
    Schema::build(KernelQuery, KernelMutation, KernelSubscription)
        .data(state)
        .finish()
}

// ============================================================================
// Axum Integration
// ============================================================================

#[cfg(feature = "axum")]
pub use axum_integration::*;

#[cfg(feature = "axum")]
mod axum_integration {
    use super::*;
    use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
    use async_graphql_axum::{GraphQLRequest, GraphQLResponse, GraphQLSubscription};
    use axum::{
        extract::State,
        response::{Html, IntoResponse},
        routing::get,
        Router,
    };

    /// GraphQL handler for HTTP POST requests.
    pub async fn graphql_handler(
        State(schema): State<KernelSchema>,
        req: GraphQLRequest,
    ) -> GraphQLResponse {
        schema.execute(req.into_inner()).await.into()
    }

    /// GraphQL Playground handler.
    pub async fn graphql_playground() -> impl IntoResponse {
        Html(playground_source(
            GraphQLPlaygroundConfig::new("/graphql").subscription_endpoint("/graphql/ws"),
        ))
    }

    /// Create an Axum router with GraphQL endpoints.
    ///
    /// Routes:
    /// - `GET /` - GraphQL Playground
    /// - `POST /` - GraphQL queries and mutations
    /// - `GET /ws` - WebSocket subscriptions
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ringkernel_ecosystem::graphql::{GraphQLState, graphql_router};
    /// use std::sync::Arc;
    ///
    /// let state = GraphQLState::new(Arc::new(handle));
    /// let app = axum::Router::new()
    ///     .nest("/graphql", graphql_router(state));
    ///
    /// let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    /// axum::serve(listener, app).await?;
    /// ```
    pub fn graphql_router(state: GraphQLState) -> Router {
        let schema = create_schema(state);

        Router::new()
            .route("/", get(graphql_playground).post(graphql_handler))
            .route_service("/ws", GraphQLSubscription::new(schema.clone()))
            .with_state(schema)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_event_from_ack() {
        let response = PersistentResponse::Ack {
            cmd_id: CommandId::new(42),
        };
        let event: KernelEvent = response.into();

        assert_eq!(event.event_type, KernelEventType::Ack);
        assert_eq!(event.command_id, Some(42));
    }

    #[test]
    fn test_kernel_event_from_progress() {
        let response = PersistentResponse::Progress {
            cmd_id: CommandId::new(1),
            current_step: 50,
            remaining: 50,
        };
        let event: KernelEvent = response.into();

        assert_eq!(event.event_type, KernelEventType::Progress);
        assert!(event.progress.is_some());

        let progress = event.progress.unwrap();
        assert_eq!(progress.completed, 50);
        assert_eq!(progress.remaining, 50);
        assert!((progress.percentage - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_kernel_event_from_error() {
        let response = PersistentResponse::Error {
            cmd_id: CommandId::new(99),
            code: 1,
            message: "Test error".to_string(),
        };
        let event: KernelEvent = response.into();

        assert_eq!(event.event_type, KernelEventType::Error);
        assert!(event.message.unwrap().contains("Test error"));
    }

    #[test]
    fn test_kernel_event_from_terminated() {
        let response = PersistentResponse::Terminated { final_step: 1000 };
        let event: KernelEvent = response.into();

        assert_eq!(event.event_type, KernelEventType::Terminated);
        assert_eq!(event.current_step, 1000);
        assert!(event.command_id.is_none());
    }

    #[test]
    fn test_stats_conversion() {
        let stats = PersistentStats {
            current_step: 1000,
            steps_remaining: 500,
            messages_processed: 500,
            total_energy: 123.456,
            k2k_sent: 10,
            k2k_received: 20,
            is_running: true,
            has_terminated: false,
            pending_commands: 5,
            pending_responses: 3,
        };

        let response: KernelStatsResponse = stats.into();
        assert_eq!(response.current_step, 1000);
        assert_eq!(response.steps_remaining, 500);
        assert_eq!(response.messages_processed, 500);
        assert_eq!(response.k2k_sent, 10);
        assert!(response.is_running);
    }
}
