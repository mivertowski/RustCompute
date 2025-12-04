//! Flow particle animation system.
//!
//! Animates particles flowing along edges to visualize money movement
//! through the accounting network.

use eframe::egui::Color32;
use nalgebra::Vector2;
use std::collections::VecDeque;

/// A particle flowing along an edge.
#[derive(Debug, Clone)]
pub struct FlowParticle {
    /// Source node index.
    pub source: u16,
    /// Target node index.
    pub target: u16,
    /// Progress along the edge (0.0 to 1.0).
    pub progress: f32,
    /// Particle speed (progress per second).
    pub speed: f32,
    /// Particle color.
    pub color: Color32,
    /// Particle size.
    pub size: f32,
    /// Whether this is a suspicious flow.
    pub suspicious: bool,
    /// Unique flow ID for grouping.
    pub flow_id: uuid::Uuid,
}

impl FlowParticle {
    /// Create a new particle.
    pub fn new(source: u16, target: u16, flow_id: uuid::Uuid) -> Self {
        Self {
            source,
            target,
            progress: 0.0,
            speed: 0.25, // Slower, more subtle movement
            color: Color32::from_rgba_unmultiplied(100, 200, 160, 150), // Semi-transparent
            size: 2.5,   // Smaller, more subtle
            suspicious: false,
            flow_id,
        }
    }

    /// Update particle position.
    pub fn update(&mut self, dt: f32) -> bool {
        self.progress += self.speed * dt;
        self.progress < 1.0
    }

    /// Get current position given source and target positions.
    pub fn position(&self, source_pos: Vector2<f32>, target_pos: Vector2<f32>) -> Vector2<f32> {
        // Ease-in-out interpolation for smoother motion
        let t = self.ease_in_out(self.progress);
        source_pos + (target_pos - source_pos) * t
    }

    /// Ease-in-out function.
    fn ease_in_out(&self, t: f32) -> f32 {
        if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
        }
    }
}

/// Manages the particle animation system.
pub struct ParticleSystem {
    /// Active particles.
    particles: VecDeque<FlowParticle>,
    /// Maximum particles.
    pub max_particles: usize,
    /// Spawn rate (particles per second per active flow).
    pub spawn_rate: f32,
    /// Time accumulator for spawning.
    spawn_accumulator: f32,
    /// Pending spawns (source, target, flow_id, color, suspicious).
    pending_spawns: Vec<(u16, u16, uuid::Uuid, Color32, bool)>,
    /// Whether animation is paused.
    pub paused: bool,
}

impl ParticleSystem {
    /// Create a new particle system.
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: VecDeque::with_capacity(max_particles),
            max_particles,
            spawn_rate: 0.8, // Reduced spawn rate for subtler animation
            spawn_accumulator: 0.0,
            pending_spawns: Vec::new(),
            paused: false,
        }
    }

    /// Queue a flow for particle spawning.
    pub fn queue_flow(
        &mut self,
        source: u16,
        target: u16,
        flow_id: uuid::Uuid,
        color: Color32,
        suspicious: bool,
    ) {
        self.pending_spawns
            .push((source, target, flow_id, color, suspicious));
    }

    /// Clear pending flows.
    pub fn clear_pending(&mut self) {
        self.pending_spawns.clear();
    }

    /// Update all particles.
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        // Remove finished particles
        self.particles.retain(|p| p.progress < 1.0);

        // Update active particles
        for particle in &mut self.particles {
            particle.update(dt);
        }

        // Spawn new particles
        self.spawn_accumulator += dt;
        let spawn_interval = 1.0 / self.spawn_rate;

        while self.spawn_accumulator >= spawn_interval && !self.pending_spawns.is_empty() {
            self.spawn_accumulator -= spawn_interval;

            // Round-robin through pending flows
            if let Some((source, target, flow_id, color, suspicious)) = self.pending_spawns.pop() {
                if self.particles.len() < self.max_particles {
                    let mut particle = FlowParticle::new(source, target, flow_id);
                    // Make particles semi-transparent for subtler effect
                    particle.color = Color32::from_rgba_unmultiplied(
                        color.r(),
                        color.g(),
                        color.b(),
                        if suspicious { 180 } else { 100 }, // More transparent for normal flows
                    );
                    particle.suspicious = suspicious;
                    particle.size = if suspicious { 4.0 } else { 2.5 }; // Smaller particles
                    particle.speed = if suspicious { 0.2 } else { 0.25 }; // Slower movement
                    self.particles.push_back(particle);
                }

                // Re-add to queue for continuous spawning
                self.pending_spawns
                    .insert(0, (source, target, flow_id, color, suspicious));
            }
        }
    }

    /// Get active particles.
    pub fn particles(&self) -> impl Iterator<Item = &FlowParticle> {
        self.particles.iter()
    }

    /// Get particle count.
    pub fn count(&self) -> usize {
        self.particles.len()
    }

    /// Clear all particles.
    pub fn clear(&mut self) {
        self.particles.clear();
        self.pending_spawns.clear();
    }

    /// Spawn a burst of particles for a specific flow.
    pub fn burst(
        &mut self,
        source: u16,
        target: u16,
        flow_id: uuid::Uuid,
        color: Color32,
        count: usize,
    ) {
        for i in 0..count {
            if self.particles.len() >= self.max_particles {
                break;
            }

            let mut particle = FlowParticle::new(source, target, flow_id);
            particle.color = color;
            particle.progress = i as f32 * 0.1; // Stagger them
            particle.speed = 0.3 + (i as f32 * 0.05); // Vary speed
            self.particles.push_back(particle);
        }
    }
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(300) // Reduced max particles for subtler effect
    }
}

/// Trail effect for particles.
#[derive(Debug, Clone)]
pub struct ParticleTrail {
    /// Trail positions.
    positions: VecDeque<Vector2<f32>>,
    /// Maximum trail length.
    max_length: usize,
    /// Trail color.
    pub color: Color32,
}

impl ParticleTrail {
    /// Create a new trail.
    pub fn new(max_length: usize, color: Color32) -> Self {
        Self {
            positions: VecDeque::with_capacity(max_length),
            max_length,
            color,
        }
    }

    /// Add a position to the trail.
    pub fn push(&mut self, position: Vector2<f32>) {
        self.positions.push_front(position);
        while self.positions.len() > self.max_length {
            self.positions.pop_back();
        }
    }

    /// Get trail positions with alpha values.
    pub fn points(&self) -> impl Iterator<Item = (Vector2<f32>, f32)> + '_ {
        let len = self.positions.len() as f32;
        self.positions.iter().enumerate().map(move |(i, &pos)| {
            let alpha = 1.0 - (i as f32 / len);
            (pos, alpha)
        })
    }

    /// Clear the trail.
    pub fn clear(&mut self) {
        self.positions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_creation() {
        let particle = FlowParticle::new(0, 1, uuid::Uuid::new_v4());
        assert_eq!(particle.progress, 0.0);
        assert_eq!(particle.source, 0);
        assert_eq!(particle.target, 1);
    }

    #[test]
    fn test_particle_update() {
        let mut particle = FlowParticle::new(0, 1, uuid::Uuid::new_v4());
        particle.speed = 1.0;
        assert!(particle.update(0.5));
        assert_eq!(particle.progress, 0.5);
        assert!(!particle.update(0.6));
    }

    #[test]
    fn test_particle_system() {
        let mut system = ParticleSystem::new(100);
        system.spawn_rate = 2.0; // Faster for test
        system.queue_flow(0, 1, uuid::Uuid::new_v4(), Color32::WHITE, false);
        system.update(1.0);
        assert!(system.count() > 0);
    }
}
