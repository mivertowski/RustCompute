//! CPU kernel implementation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use tokio::sync::{mpsc, Notify};

use ringkernel_core::control::ControlBlock;
use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::hlc::HlcClock;
use ringkernel_core::k2k::{DeliveryReceipt, K2KEndpoint, K2KMessage};
use ringkernel_core::message::{CorrelationId, MessageEnvelope};
use ringkernel_core::queue::{BoundedQueue, MessageQueue};
use ringkernel_core::runtime::{
    KernelHandle, KernelHandleInner, KernelId, KernelState, KernelStatus, LaunchOptions,
};
use ringkernel_core::telemetry::{KernelMetrics, TelemetryBuffer};

/// CPU-based kernel implementation.
pub struct CpuKernel {
    /// Kernel identifier.
    id: KernelId,
    /// Numeric ID for message routing.
    id_num: u64,
    /// Current state.
    state: RwLock<KernelState>,
    /// Launch options.
    options: LaunchOptions,
    /// Control block.
    control: RwLock<ControlBlock>,
    /// Telemetry buffer.
    telemetry: RwLock<TelemetryBuffer>,
    /// Input queue (host -> kernel).
    input_queue: Arc<BoundedQueue>,
    /// Output queue (kernel -> host).
    output_queue: Arc<BoundedQueue>,
    /// HLC clock.
    clock: Arc<HlcClock>,
    /// Correlation waiters.
    correlation_waiters: Mutex<std::collections::HashMap<u64, mpsc::Sender<MessageEnvelope>>>,
    /// Termination notifier.
    terminate_notify: Notify,
    /// Launch time.
    launched_at: Instant,
    /// Message counter.
    message_counter: AtomicU64,
    /// K2K endpoint for kernel-to-kernel messaging.
    k2k_endpoint: Mutex<Option<K2KEndpoint>>,
}

impl CpuKernel {
    /// Create a new CPU kernel.
    pub fn new(id: KernelId, options: LaunchOptions, node_id: u64) -> Self {
        Self::new_with_k2k(id, options, node_id, None)
    }

    /// Create a new CPU kernel with optional K2K endpoint.
    pub fn new_with_k2k(
        id: KernelId,
        options: LaunchOptions,
        node_id: u64,
        k2k_endpoint: Option<K2KEndpoint>,
    ) -> Self {
        static KERNEL_COUNTER: AtomicU64 = AtomicU64::new(1);
        let id_num = KERNEL_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Save capacity values before moving options
        let input_capacity = options.input_queue_capacity;
        let output_capacity = options.output_queue_capacity;

        let control = ControlBlock::with_capacities(input_capacity as u32, output_capacity as u32);

        Self {
            id,
            id_num,
            state: RwLock::new(KernelState::Created),
            options,
            control: RwLock::new(control),
            telemetry: RwLock::new(TelemetryBuffer::new()),
            input_queue: Arc::new(BoundedQueue::new(input_capacity)),
            output_queue: Arc::new(BoundedQueue::new(output_capacity)),
            clock: Arc::new(HlcClock::new(node_id)),
            correlation_waiters: Mutex::new(std::collections::HashMap::new()),
            terminate_notify: Notify::new(),
            launched_at: Instant::now(),
            message_counter: AtomicU64::new(0),
            k2k_endpoint: Mutex::new(k2k_endpoint),
        }
    }

    /// Launch the kernel (start processing).
    pub fn launch(&self) {
        let mut state = self.state.write();
        if *state == KernelState::Created {
            *state = KernelState::Launched;
        }
    }

    /// Get kernel ID.
    pub fn id(&self) -> &KernelId {
        &self.id
    }

    /// Get current state.
    pub fn state(&self) -> KernelState {
        *self.state.read()
    }

    /// Process one message (for testing).
    pub fn process_message(&self, envelope: MessageEnvelope) -> Result<()> {
        // Update telemetry
        let mut telemetry = self.telemetry.write();
        telemetry.messages_processed += 1;

        // For CPU backend, we just pass messages through
        // In a real implementation, this would call the registered handler
        self.output_queue.try_enqueue(envelope)?;

        Ok(())
    }

    /// Create a handle to this kernel.
    pub fn handle(self: &Arc<Self>) -> KernelHandle {
        KernelHandle::new(
            self.id.clone(),
            Arc::clone(self) as Arc<dyn KernelHandleInner>,
        )
    }

    /// Check if K2K messaging is enabled for this kernel.
    pub fn is_k2k_enabled(&self) -> bool {
        self.k2k_endpoint.lock().is_some()
    }

    /// Send a K2K message to another kernel.
    pub async fn k2k_send(
        &self,
        destination: KernelId,
        envelope: MessageEnvelope,
    ) -> Result<DeliveryReceipt> {
        let endpoint = {
            let mut endpoint_guard = self.k2k_endpoint.lock();
            endpoint_guard.take().ok_or_else(|| {
                RingKernelError::K2KError("K2K not enabled for this kernel".to_string())
            })?
        };
        let result = endpoint.send(destination, envelope).await;
        // Put it back
        *self.k2k_endpoint.lock() = Some(endpoint);
        result
    }

    /// Try to receive a K2K message (non-blocking).
    pub fn k2k_try_recv(&self) -> Option<K2KMessage> {
        let mut endpoint_guard = self.k2k_endpoint.lock();
        endpoint_guard.as_mut()?.try_receive()
    }

    /// Receive a K2K message (blocking).
    pub async fn k2k_recv(&self) -> Option<K2KMessage> {
        // We need to take the endpoint out temporarily to use the async receiver
        let mut endpoint = {
            let mut endpoint_guard = self.k2k_endpoint.lock();
            endpoint_guard.take()?
        };
        let result = endpoint.receive().await;
        // Put it back
        *self.k2k_endpoint.lock() = Some(endpoint);
        result
    }
}

#[async_trait]
impl KernelHandleInner for CpuKernel {
    fn kernel_id_num(&self) -> u64 {
        self.id_num
    }

    fn current_timestamp(&self) -> ringkernel_core::hlc::HlcTimestamp {
        self.clock.now()
    }

    async fn activate(&self) -> Result<()> {
        let mut state = self.state.write();
        if !state.can_activate() {
            return Err(RingKernelError::InvalidStateTransition {
                from: format!("{:?}", *state),
                to: "Active".to_string(),
            });
        }
        *state = KernelState::Active;

        // Update control block
        let mut control = self.control.write();
        control.is_active = 1;

        Ok(())
    }

    async fn deactivate(&self) -> Result<()> {
        let mut state = self.state.write();
        if !state.can_deactivate() {
            return Err(RingKernelError::InvalidStateTransition {
                from: format!("{:?}", *state),
                to: "Deactivated".to_string(),
            });
        }
        *state = KernelState::Deactivated;

        // Update control block
        let mut control = self.control.write();
        control.is_active = 0;

        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        let mut state = self.state.write();
        if !state.can_terminate() {
            return Err(RingKernelError::InvalidStateTransition {
                from: format!("{:?}", *state),
                to: "Terminated".to_string(),
            });
        }
        *state = KernelState::Terminating;

        // Update control block
        {
            let mut control = self.control.write();
            control.should_terminate = 1;
            control.is_active = 0;
        }

        // Notify waiting tasks
        self.terminate_notify.notify_waiters();

        // Mark as terminated
        *state = KernelState::Terminated;
        {
            let mut control = self.control.write();
            control.has_terminated = 1;
        }

        Ok(())
    }

    async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()> {
        let state = self.state();
        if !state.is_running() {
            return Err(RingKernelError::KernelNotActive(self.id.to_string()));
        }

        self.input_queue
            .enqueue_timeout(envelope, Duration::from_secs(5))?;
        self.message_counter.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    async fn receive(&self) -> Result<MessageEnvelope> {
        self.receive_timeout(Duration::from_secs(30)).await
    }

    async fn receive_timeout(&self, timeout: Duration) -> Result<MessageEnvelope> {
        let envelope = self.output_queue.dequeue_timeout(timeout)?;

        // Check if this matches any correlation waiter
        if envelope.header.correlation_id.is_some() {
            let waiters = self.correlation_waiters.lock();
            if let Some(sender) = waiters.get(&envelope.header.correlation_id.0) {
                let _ = sender.try_send(envelope.clone());
            }
        }

        Ok(envelope)
    }

    fn try_receive(&self) -> Result<MessageEnvelope> {
        self.output_queue.try_dequeue()
    }

    async fn receive_correlated(
        &self,
        correlation: CorrelationId,
        timeout: Duration,
    ) -> Result<MessageEnvelope> {
        let (tx, mut rx) = mpsc::channel(1);

        // Register waiter
        {
            let mut waiters = self.correlation_waiters.lock();
            waiters.insert(correlation.0, tx);
        }

        // Wait for response
        let result = tokio::time::timeout(timeout, rx.recv()).await;

        // Cleanup waiter
        {
            let mut waiters = self.correlation_waiters.lock();
            waiters.remove(&correlation.0);
        }

        match result {
            Ok(Some(envelope)) => Ok(envelope),
            Ok(None) => Err(RingKernelError::ChannelClosed),
            Err(_) => Err(RingKernelError::Timeout(timeout)),
        }
    }

    fn status(&self) -> KernelStatus {
        let state = *self.state.read();
        let control = self.control.read();

        KernelStatus {
            id: self.id.clone(),
            state,
            mode: self.options.mode,
            input_queue_depth: self.input_queue.len(),
            output_queue_depth: self.output_queue.len(),
            messages_processed: control.messages_processed,
            uptime: self.launched_at.elapsed(),
        }
    }

    fn metrics(&self) -> KernelMetrics {
        let telemetry = *self.telemetry.read();
        KernelMetrics {
            telemetry,
            kernel_id: self.id.to_string(),
            collected_at: Instant::now(),
            uptime: self.launched_at.elapsed(),
            invocations: 0,
            bytes_to_device: 0,
            bytes_from_device: 0,
            gpu_memory_used: 0,
            host_memory_used: 0,
        }
    }

    async fn wait(&self) -> Result<()> {
        loop {
            if self.state().is_finished() {
                return Ok(());
            }
            self.terminate_notify.notified().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringkernel_core::hlc::HlcTimestamp;
    use ringkernel_core::message::MessageHeader;

    fn make_envelope() -> MessageEnvelope {
        MessageEnvelope {
            header: MessageHeader::new(1, 0, 1, 8, HlcTimestamp::now(1)),
            payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
        }
    }

    #[tokio::test]
    async fn test_kernel_lifecycle() {
        let kernel = Arc::new(CpuKernel::new(
            KernelId::new("test"),
            LaunchOptions::default(),
            1,
        ));

        assert_eq!(kernel.state(), KernelState::Created);

        kernel.launch();
        assert_eq!(kernel.state(), KernelState::Launched);

        kernel.activate().await.unwrap();
        assert_eq!(kernel.state(), KernelState::Active);

        kernel.deactivate().await.unwrap();
        assert_eq!(kernel.state(), KernelState::Deactivated);

        kernel.activate().await.unwrap();
        assert_eq!(kernel.state(), KernelState::Active);

        kernel.terminate().await.unwrap();
        assert_eq!(kernel.state(), KernelState::Terminated);
    }

    #[tokio::test]
    async fn test_send_receive() {
        let kernel = Arc::new(CpuKernel::new(
            KernelId::new("test"),
            LaunchOptions::default(),
            1,
        ));

        kernel.launch();
        kernel.activate().await.unwrap();

        // Send message
        let env = make_envelope();
        kernel.send_envelope(env.clone()).await.unwrap();

        // Process it (simulates kernel processing)
        let recv = kernel.input_queue.try_dequeue().unwrap();
        kernel.output_queue.try_enqueue(recv).unwrap();

        // Receive
        let result = kernel.try_receive().unwrap();
        assert_eq!(result.header.message_type, env.header.message_type);
    }

    #[tokio::test]
    async fn test_status() {
        let kernel = Arc::new(CpuKernel::new(
            KernelId::new("test"),
            LaunchOptions::default(),
            1,
        ));

        kernel.launch();
        kernel.activate().await.unwrap();

        let status = kernel.status();
        assert_eq!(status.id.as_str(), "test");
        assert_eq!(status.state, KernelState::Active);
    }
}
