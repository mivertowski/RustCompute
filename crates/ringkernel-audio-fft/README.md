# ringkernel-audio-fft

GPU-accelerated audio FFT processing with direct/ambience signal separation.

## Overview

This crate demonstrates RingKernel's actor model by implementing a novel audio processing pipeline where each FFT frequency bin is an independent GPU actor. Actors communicate with neighbors via K2K (kernel-to-kernel) messaging to perform coherence analysis and separate direct sound from room ambience.

## Architecture

```
Audio Input -> FFT -> [Bin Actors with K2K] -> Separation -> Mixer -> IFFT -> Output
                           |   |   |
                         neighbor links
```

Each frequency bin actor:
1. Receives its FFT data (magnitude + phase)
2. Exchanges data with left/right neighbors via K2K
3. Computes coherence metrics
4. Separates into direct and ambient components

## Signal Separation Algorithm

The separation is based on inter-bin phase coherence:

- **Direct signal**: High phase coherence between neighboring bins (transient, localized sounds)
- **Ambience**: Low phase coherence, diffuse energy distribution (reverb, room tone)

## Usage

```rust
use ringkernel_audio_fft::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let processor = AudioFftProcessor::builder()
        .fft_size(2048)
        .hop_size(512)
        .sample_rate(44100)
        .build()
        .await?;

    let input = AudioInput::from_file("input.wav")?;

    let output = processor
        .process(input)
        .dry_wet(0.5)       // 50% wet (ambience)
        .gain_boost(1.2)    // 20% gain boost
        .run()
        .await?;

    output.direct.write_to_file("direct.wav")?;
    output.ambience.write_to_file("ambience.wav")?;

    Ok(())
}
```

## Components

| Module | Description |
|--------|-------------|
| `audio_input` | Audio file I/O and device streaming |
| `bin_actor` | FFT bin actor implementation |
| `fft` | FFT/IFFT processing with window functions |
| `separation` | Coherence analysis and signal separation |
| `mixer` | Dry/wet mixing and gain control |
| `processor` | High-level processing pipeline |

## Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_size` | 2048 | FFT window size (power of 2) |
| `hop_size` | 512 | Overlap between windows |
| `sample_rate` | 44100 | Audio sample rate |
| `dry_wet` | 0.5 | Mix ratio (0 = direct only, 1 = ambience only) |

## Testing

```bash
cargo test -p ringkernel-audio-fft
```

The crate includes 32 tests covering FFT processing, coherence analysis, and signal separation.

## License

Apache-2.0
