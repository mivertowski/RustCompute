//! FFT bin actors with K2K neighbor messaging.
//!
//! Each frequency bin is represented as an independent GPU actor that:
//! 1. Receives FFT bin data from the host
//! 2. Exchanges information with neighboring bins via K2K messaging
//! 3. Performs coherence analysis for direct/ambience separation
//! 4. Outputs separated bin data

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use ringkernel_core::prelude::*;
use ringkernel_core::k2k::K2KStats;

use crate::error::{AudioFftError, Result};
use crate::messages::{Complex, FrequencyBin, NeighborData, SeparatedBin};
use crate::separation::{CoherenceAnalyzer, SeparationConfig};

/// State maintained by each bin actor.
#[derive(Debug, Clone)]
pub struct BinActorState {
    /// Bin index.
    pub bin_index: u32,
    /// Current frame ID.
    pub current_frame: u64,
    /// Current bin value.
    pub current_value: Complex,
    /// Previous frame value.
    pub prev_value: Option<Complex>,
    /// Left neighbor data (lower frequency).
    pub left_neighbor: Option<NeighborData>,
    /// Right neighbor data (higher frequency).
    pub right_neighbor: Option<NeighborData>,
    /// Computed coherence with neighbors.
    pub coherence: f32,
    /// Temporal smoothing state.
    pub smoothed_coherence: f32,
    /// Phase derivative (for transient detection).
    pub phase_derivative: f32,
    /// Spectral flux.
    pub spectral_flux: f32,
}

impl BinActorState {
    /// Create a new bin actor state.
    pub fn new(bin_index: u32) -> Self {
        Self {
            bin_index,
            current_frame: 0,
            current_value: Complex::default(),
            prev_value: None,
            left_neighbor: None,
            right_neighbor: None,
            coherence: 0.5,
            smoothed_coherence: 0.5,
            phase_derivative: 0.0,
            spectral_flux: 0.0,
        }
    }

    /// Update with new bin data.
    pub fn update(&mut self, bin: &FrequencyBin) {
        self.prev_value = Some(self.current_value);
        self.current_value = bin.value;
        self.current_frame = bin.frame_id;

        // Calculate phase derivative
        if let Some(prev) = self.prev_value {
            let prev_phase = prev.phase();
            let curr_phase = self.current_value.phase();
            // Unwrap phase difference
            let mut phase_diff = curr_phase - prev_phase;
            while phase_diff > std::f32::consts::PI {
                phase_diff -= 2.0 * std::f32::consts::PI;
            }
            while phase_diff < -std::f32::consts::PI {
                phase_diff += 2.0 * std::f32::consts::PI;
            }
            self.phase_derivative = phase_diff;

            // Calculate spectral flux
            let prev_mag = prev.magnitude();
            let curr_mag = self.current_value.magnitude();
            self.spectral_flux = (curr_mag - prev_mag).max(0.0); // Only onset (positive flux)
        }

        // Clear neighbor data for new frame
        self.left_neighbor = None;
        self.right_neighbor = None;
    }

    /// Set neighbor data.
    pub fn set_neighbor(&mut self, data: NeighborData, is_left: bool) {
        if is_left {
            self.left_neighbor = Some(data);
        } else {
            self.right_neighbor = Some(data);
        }
    }

    /// Check if we have all neighbor data.
    pub fn has_all_neighbors(&self, has_left: bool, has_right: bool) -> bool {
        (!has_left || self.left_neighbor.is_some()) && (!has_right || self.right_neighbor.is_some())
    }

    /// Create neighbor data to send to adjacent bins.
    pub fn to_neighbor_data(&self) -> NeighborData {
        NeighborData {
            source_bin: self.bin_index,
            frame_id: self.current_frame,
            value: self.current_value,
            magnitude: self.current_value.magnitude(),
            phase: self.current_value.phase(),
            phase_derivative: self.phase_derivative,
            spectral_flux: self.spectral_flux,
        }
    }
}

/// Handle to a single bin actor.
pub struct BinActorHandle {
    /// Bin index.
    pub bin_index: u32,
    /// Kernel ID.
    kernel_id: KernelId,
    /// K2K endpoint for this actor.
    endpoint: K2KEndpoint,
    /// State (shared for monitoring).
    state: Arc<RwLock<BinActorState>>,
    /// Input channel for bin data.
    input_tx: mpsc::Sender<FrequencyBin>,
    /// Output channel for separated data.
    output_rx: mpsc::Receiver<SeparatedBin>,
    /// Running flag.
    running: Arc<AtomicBool>,
}

impl BinActorHandle {
    /// Send bin data to the actor.
    pub async fn send_bin(&self, bin: FrequencyBin) -> Result<()> {
        self.input_tx
            .send(bin)
            .await
            .map_err(|e| AudioFftError::kernel(format!("Failed to send bin data: {}", e)))
    }

    /// Receive separated bin data.
    pub async fn receive_separated(&mut self) -> Option<SeparatedBin> {
        self.output_rx.recv().await
    }

    /// Get the current state.
    pub fn state(&self) -> BinActorState {
        self.state.read().clone()
    }

    /// Get the kernel ID.
    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop the actor.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// A single bin actor that processes one frequency bin.
pub struct BinActor {
    /// Bin index.
    bin_index: u32,
    /// Total number of bins.
    total_bins: u32,
    /// Kernel ID.
    kernel_id: KernelId,
    /// State.
    state: Arc<RwLock<BinActorState>>,
    /// K2K endpoint.
    endpoint: K2KEndpoint,
    /// Left neighbor kernel ID (if any).
    left_neighbor_id: Option<KernelId>,
    /// Right neighbor kernel ID (if any).
    right_neighbor_id: Option<KernelId>,
    /// Input channel.
    input_rx: mpsc::Receiver<FrequencyBin>,
    /// Output channel.
    output_tx: mpsc::Sender<SeparatedBin>,
    /// Coherence analyzer.
    analyzer: CoherenceAnalyzer,
    /// Separation config.
    config: SeparationConfig,
    /// Running flag.
    running: Arc<AtomicBool>,
    /// Frame counter.
    frame_counter: AtomicU64,
}

impl BinActor {
    /// Create a new bin actor.
    pub fn new(
        bin_index: u32,
        total_bins: u32,
        broker: &Arc<K2KBroker>,
        config: SeparationConfig,
    ) -> (Self, BinActorHandle) {
        let kernel_id = KernelId::new(format!("bin_actor_{}", bin_index));
        let endpoint = broker.register(kernel_id.clone());

        let state = Arc::new(RwLock::new(BinActorState::new(bin_index)));
        let running = Arc::new(AtomicBool::new(true));

        let (input_tx, input_rx) = mpsc::channel(64);
        let (output_tx, output_rx) = mpsc::channel(64);

        // Create handle's endpoint separately
        let handle_endpoint = broker.register(KernelId::new(format!("bin_actor_{}_handle", bin_index)));

        let handle = BinActorHandle {
            bin_index,
            kernel_id: kernel_id.clone(),
            endpoint: handle_endpoint,
            state: state.clone(),
            input_tx,
            output_rx,
            running: running.clone(),
        };

        let actor = Self {
            bin_index,
            total_bins,
            kernel_id,
            state,
            endpoint,
            left_neighbor_id: None,
            right_neighbor_id: None,
            input_rx,
            output_tx,
            analyzer: CoherenceAnalyzer::new(config.clone()),
            config,
            running,
            frame_counter: AtomicU64::new(0),
        };

        (actor, handle)
    }

    /// Set neighbor kernel IDs.
    pub fn set_neighbors(&mut self, left: Option<KernelId>, right: Option<KernelId>) {
        self.left_neighbor_id = left;
        self.right_neighbor_id = right;
    }

    /// Run the actor processing loop.
    pub async fn run(&mut self) -> Result<()> {
        info!("Bin actor {} starting", self.bin_index);

        while self.running.load(Ordering::Relaxed) {
            // Wait for input bin data
            let bin = match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                self.input_rx.recv(),
            )
            .await
            {
                Ok(Some(bin)) => bin,
                Ok(None) => {
                    // Channel closed
                    break;
                }
                Err(_) => {
                    // Timeout, check if still running
                    continue;
                }
            };

            trace!("Bin {} processing frame {}", self.bin_index, bin.frame_id);

            // Update state
            {
                let mut state = self.state.write();
                state.update(&bin);
            }

            // Send neighbor data via K2K
            self.send_neighbor_data().await?;

            // Receive neighbor data via K2K
            self.receive_neighbor_data().await?;

            // Perform separation
            let separated = self.compute_separation();

            // Send output
            if self.output_tx.send(separated).await.is_err() {
                warn!("Output channel closed for bin {}", self.bin_index);
                break;
            }

            self.frame_counter.fetch_add(1, Ordering::Relaxed);
        }

        info!("Bin actor {} stopped", self.bin_index);
        Ok(())
    }

    /// Send neighbor data to adjacent bins.
    async fn send_neighbor_data(&mut self) -> Result<()> {
        let neighbor_data = self.state.read().to_neighbor_data();

        // Send to left neighbor
        if let Some(left_id) = &self.left_neighbor_id {
            let envelope = MessageEnvelope::new(
                &neighbor_data,
                self.bin_index as u64,
                (self.bin_index - 1) as u64,
                HlcTimestamp::now(self.bin_index as u64),
            );

            match self.endpoint.send(left_id.clone(), envelope).await {
                Ok(receipt) if receipt.status == DeliveryStatus::Delivered => {
                    trace!("Sent to left neighbor {}", left_id);
                }
                Ok(receipt) => {
                    trace!("Left neighbor delivery status: {:?}", receipt.status);
                }
                Err(e) => {
                    trace!("Failed to send to left neighbor: {}", e);
                }
            }
        }

        // Send to right neighbor
        if let Some(right_id) = &self.right_neighbor_id {
            let envelope = MessageEnvelope::new(
                &neighbor_data,
                self.bin_index as u64,
                (self.bin_index + 1) as u64,
                HlcTimestamp::now(self.bin_index as u64),
            );

            match self.endpoint.send(right_id.clone(), envelope).await {
                Ok(receipt) if receipt.status == DeliveryStatus::Delivered => {
                    trace!("Sent to right neighbor {}", right_id);
                }
                Ok(receipt) => {
                    trace!("Right neighbor delivery status: {:?}", receipt.status);
                }
                Err(e) => {
                    trace!("Failed to send to right neighbor: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Receive neighbor data from adjacent bins.
    async fn receive_neighbor_data(&mut self) -> Result<()> {
        let has_left = self.left_neighbor_id.is_some();
        let has_right = self.right_neighbor_id.is_some();

        // Try to receive with a short timeout
        let timeout = std::time::Duration::from_millis(10);
        let deadline = std::time::Instant::now() + timeout;

        while std::time::Instant::now() < deadline {
            match self.endpoint.try_receive() {
                Some(k2k_msg) => {
                    // Deserialize neighbor data
                    if let Ok(neighbor_data) = NeighborData::deserialize(&k2k_msg.envelope.payload)
                    {
                        let is_left = neighbor_data.source_bin < self.bin_index;
                        let mut state = self.state.write();
                        state.set_neighbor(neighbor_data, is_left);

                        if state.has_all_neighbors(has_left, has_right) {
                            break;
                        }
                    }
                }
                None => {
                    // No message available, brief yield
                    tokio::task::yield_now().await;
                }
            }
        }

        Ok(())
    }

    /// Compute the separation of direct and ambient signals.
    fn compute_separation(&mut self) -> SeparatedBin {
        let state = self.state.read();

        // Compute coherence based on neighbor data
        let (coherence, transient) = self.analyzer.analyze(
            &state.current_value,
            state.left_neighbor.as_ref(),
            state.right_neighbor.as_ref(),
            state.phase_derivative,
            state.spectral_flux,
        );

        // Update smoothed coherence
        drop(state);
        {
            let mut state = self.state.write();
            state.coherence = coherence;
            state.smoothed_coherence = state.smoothed_coherence * self.config.temporal_smoothing
                + coherence * (1.0 - self.config.temporal_smoothing);
        }

        let state = self.state.read();
        let smoothed = state.smoothed_coherence;

        // Separate direct and ambient components
        let direct_ratio = smoothed.powf(self.config.separation_curve);
        let ambient_ratio = 1.0 - direct_ratio;

        let direct = state.current_value.scale(direct_ratio);
        let ambience = state.current_value.scale(ambient_ratio);

        SeparatedBin::new(
            state.current_frame,
            self.bin_index,
            direct,
            ambience,
            smoothed,
            transient,
        )
    }
}

/// Network of bin actors with K2K messaging.
pub struct BinNetwork {
    /// Number of bins.
    num_bins: usize,
    /// K2K broker.
    broker: Arc<K2KBroker>,
    /// Actor handles.
    handles: Vec<BinActorHandle>,
    /// Actor tasks.
    tasks: Vec<tokio::task::JoinHandle<Result<()>>>,
    /// Configuration.
    config: SeparationConfig,
    /// Running flag.
    running: Arc<AtomicBool>,
}

impl BinNetwork {
    /// Create a new bin network.
    pub async fn new(num_bins: usize, config: SeparationConfig) -> Result<Self> {
        info!("Creating bin network with {} bins", num_bins);

        let broker = K2KBuilder::new()
            .max_pending_messages(num_bins * 4)
            .delivery_timeout_ms(100)
            .build();

        let mut actors: Vec<BinActor> = Vec::with_capacity(num_bins);
        let mut handles: Vec<BinActorHandle> = Vec::with_capacity(num_bins);

        // Create all actors
        for i in 0..num_bins {
            let (actor, handle) = BinActor::new(i as u32, num_bins as u32, &broker, config.clone());
            actors.push(actor);
            handles.push(handle);
        }

        // Set up neighbor relationships
        for i in 0..num_bins {
            let left = if i > 0 {
                Some(KernelId::new(format!("bin_actor_{}", i - 1)))
            } else {
                None
            };
            let right = if i < num_bins - 1 {
                Some(KernelId::new(format!("bin_actor_{}", i + 1)))
            } else {
                None
            };
            actors[i].set_neighbors(left, right);
        }

        let running = Arc::new(AtomicBool::new(true));

        // Spawn actor tasks
        let mut tasks = Vec::with_capacity(num_bins);
        for mut actor in actors {
            let task = tokio::spawn(async move { actor.run().await });
            tasks.push(task);
        }

        Ok(Self {
            num_bins,
            broker,
            handles,
            tasks,
            config,
            running,
        })
    }

    /// Get the number of bins.
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Get a handle to a specific bin.
    pub fn get_handle(&self, bin_index: usize) -> Option<&BinActorHandle> {
        self.handles.get(bin_index)
    }

    /// Send bin data to all actors.
    pub async fn send_bins(&self, bins: &[FrequencyBin]) -> Result<()> {
        for (i, bin) in bins.iter().enumerate() {
            if i < self.handles.len() {
                self.handles[i].send_bin(bin.clone()).await?;
            }
        }
        Ok(())
    }

    /// Receive separated bins from all actors.
    pub async fn receive_separated(&mut self) -> Result<Vec<SeparatedBin>> {
        let mut results = Vec::with_capacity(self.num_bins);

        for handle in &mut self.handles {
            if let Some(separated) = handle.receive_separated().await {
                results.push(separated);
            }
        }

        // Sort by bin index
        results.sort_by_key(|b| b.bin_index);

        Ok(results)
    }

    /// Process a frame of FFT bins and return separated bins.
    pub async fn process_frame(
        &mut self,
        frame_id: u64,
        bins: &[Complex],
        sample_rate: u32,
        fft_size: usize,
    ) -> Result<Vec<SeparatedBin>> {
        // Convert to FrequencyBin messages
        let freq_bins: Vec<FrequencyBin> = bins
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let frequency_hz = i as f32 * sample_rate as f32 / fft_size as f32;
                FrequencyBin::new(frame_id, i as u32, bins.len() as u32, value, frequency_hz)
            })
            .collect();

        // Send to actors
        self.send_bins(&freq_bins).await?;

        // Receive separated results
        self.receive_separated().await
    }

    /// Stop all actors.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping bin network");
        self.running.store(false, Ordering::Relaxed);

        for handle in &self.handles {
            handle.stop();
        }

        // Wait for tasks to complete
        for task in self.tasks.drain(..) {
            let _ = task.await;
        }

        Ok(())
    }

    /// Get K2K broker statistics.
    pub fn k2k_stats(&self) -> K2KStats {
        self.broker.stats()
    }
}

impl Drop for BinNetwork {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        for handle in &self.handles {
            handle.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bin_network_creation() {
        let config = SeparationConfig::default();
        let network = BinNetwork::new(16, config).await.unwrap();

        assert_eq!(network.num_bins(), 16);

        let stats = network.k2k_stats();
        assert!(stats.registered_endpoints >= 16);
    }

    #[test]
    fn test_bin_actor_state() {
        let mut state = BinActorState::new(5);
        assert_eq!(state.bin_index, 5);
        assert_eq!(state.coherence, 0.5);

        let bin = FrequencyBin::new(1, 5, 1024, Complex::new(1.0, 0.0), 440.0);
        state.update(&bin);

        assert_eq!(state.current_frame, 1);
        assert!((state.current_value.magnitude() - 1.0).abs() < 1e-6);
    }
}
