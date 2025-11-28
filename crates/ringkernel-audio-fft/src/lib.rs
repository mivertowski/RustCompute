//! # RingKernel Audio FFT
//!
//! GPU-accelerated audio FFT processing with direct/ambience signal separation
//! using the RingKernel persistent actor model.
//!
//! ## Architecture
//!
//! This crate implements a novel approach to audio processing where each FFT frequency
//! bin is represented as an independent GPU actor. These actors communicate with their
//! neighbors via K2K (kernel-to-kernel) messaging to perform coherence analysis and
//! separate direct sound from room ambience.
//!
//! ```text
//! Audio Input -> FFT -> [Bin Actors with K2K] -> Separation -> Mixer -> IFFT -> Output
//!                            |   |   |
//!                          neighbor links
//! ```
//!
//! ## Signal Separation Algorithm
//!
//! The separation is based on inter-bin coherence analysis:
//! - **Direct signal**: High phase coherence between neighboring bins (transient, localized)
//! - **Ambience**: Low phase coherence, diffuse energy distribution
//!
//! Each bin actor:
//! 1. Receives its FFT data (magnitude + phase)
//! 2. Exchanges data with left/right neighbors via K2K
//! 3. Computes coherence metrics
//! 4. Separates into direct and ambient components
//!
//! ## Example
//!
//! ```ignore
//! use ringkernel_audio_fft::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create the audio FFT processor
//!     let processor = AudioFftProcessor::builder()
//!         .fft_size(2048)
//!         .hop_size(512)
//!         .sample_rate(44100)
//!         .build()
//!         .await?;
//!
//!     // Load audio file
//!     let input = AudioInput::from_file("input.wav")?;
//!
//!     // Process with dry/wet control
//!     let output = processor
//!         .process(input)
//!         .dry_wet(0.5)       // 50% wet (ambience)
//!         .gain_boost(1.2)    // 20% gain boost
//!         .run()
//!         .await?;
//!
//!     // Write output
//!     output.direct.write_to_file("direct.wav")?;
//!     output.ambience.write_to_file("ambience.wav")?;
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod audio_input;
pub mod bin_actor;
pub mod error;
pub mod fft;
pub mod messages;
pub mod mixer;
pub mod processor;
pub mod separation;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::audio_input::{AudioInput, AudioOutput, AudioSource, DeviceStream, FileSource};
    pub use crate::bin_actor::{BinActor, BinActorHandle, BinNetwork};
    pub use crate::error::{AudioFftError, Result};
    pub use crate::fft::{FftProcessor, IfftProcessor, WindowFunction};
    pub use crate::messages::*;
    pub use crate::mixer::{DryWetMixer, MixerConfig};
    pub use crate::processor::{AudioFftProcessor, AudioFftProcessorBuilder, ProcessingOutput};
    pub use crate::separation::{CoherenceAnalyzer, SeparationConfig, SignalSeparator};
}

// Re-exports
pub use error::{AudioFftError, Result};
pub use processor::{AudioFftProcessor, AudioFftProcessorBuilder, ProcessingOutput};
