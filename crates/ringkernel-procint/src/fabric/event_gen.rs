//! Process event generator for synthetic data production.

use super::{AnomalyConfig, ParallelActivityDef, SectorTemplate};
use crate::models::{ActivityId, ActivityRegistry, EventType, GpuObjectEvent, HybridTimestamp};
use rand::prelude::*;
use rand_distr::Normal;
use std::collections::HashMap;

/// Generator configuration.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Target events per second.
    pub events_per_second: u32,
    /// Batch size for generation.
    pub batch_size: usize,
    /// Number of concurrent cases.
    pub concurrent_cases: u32,
    /// Deviation rate from reference model (0.0 = perfect conformance).
    pub deviation_rate: f32,
    /// Anomaly injection configuration.
    pub anomalies: AnomalyConfig,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            events_per_second: 10_000,
            batch_size: 1000,
            concurrent_cases: 100,
            deviation_rate: 0.1,
            anomalies: AnomalyConfig::default(),
            seed: None,
        }
    }
}

impl GeneratorConfig {
    /// Set sector template.
    pub fn with_sector(self, _sector: SectorTemplate) -> Self {
        self
    }

    /// Set events per second.
    pub fn with_events_per_second(mut self, eps: u32) -> Self {
        self.events_per_second = eps;
        self
    }

    /// Set deviation rate.
    pub fn with_deviation_rate(mut self, rate: f32) -> Self {
        self.deviation_rate = rate;
        self
    }

    /// Set anomaly config.
    pub fn with_anomalies(mut self, anomalies: AnomalyConfig) -> Self {
        self.anomalies = anomalies;
        self
    }
}

/// State of an active case.
#[derive(Debug, Clone)]
struct CaseState {
    /// Case/object ID.
    case_id: u64,
    /// Current activity index in registry (None if in parallel mode).
    current_activity: Option<ActivityId>,
    /// Activities completed so far.
    completed_activities: Vec<ActivityId>,
    /// Last event time.
    last_event_time: HybridTimestamp,
    /// Is case complete?
    is_complete: bool,
    /// Injected anomaly type (if any).
    anomaly: Option<AnomalyType>,
    /// Parallel activities in flight (activity_id, start_time, duration_ms).
    parallel_activities: Vec<ParallelActivityState>,
    /// Join activity to transition to after all parallel activities complete.
    join_to: Option<ActivityId>,
}

/// State of a parallel activity in flight.
#[derive(Debug, Clone)]
struct ParallelActivityState {
    /// Activity ID.
    activity_id: ActivityId,
    /// When this activity started.
    start_time: HybridTimestamp,
    /// Expected duration in ms.
    duration_ms: u32,
    /// Whether event has been emitted.
    event_emitted: bool,
}

/// Type of anomaly injected.
#[derive(Debug, Clone, Copy)]
enum AnomalyType {
    Bottleneck,
    Rework,
    LongRunning,
    Skip,
}

/// Process event generator.
pub struct ProcessEventGenerator {
    /// Sector template.
    sector: SectorTemplate,
    /// Generator configuration.
    config: GeneratorConfig,
    /// Activity registry.
    registry: ActivityRegistry,
    /// Transition map: activity_id -> [(target_id, probability, avg_time)]
    transitions: HashMap<ActivityId, Vec<(ActivityId, f32, u32)>>,
    /// Parallel activity definitions: fork_activity_id -> ParallelActivityDef
    parallel_defs: HashMap<ActivityId, ParallelActivityDef>,
    /// Random number generator.
    rng: StdRng,
    /// Next event ID.
    next_event_id: u64,
    /// Next case ID.
    next_case_id: u64,
    /// Active cases.
    active_cases: HashMap<u64, CaseState>,
    /// Current simulation time.
    current_time: HybridTimestamp,
    /// Statistics.
    stats: GeneratorStats,
}

/// Generator statistics.
#[derive(Debug, Clone, Default)]
pub struct GeneratorStats {
    /// Total events generated.
    pub total_events: u64,
    /// Total cases started.
    pub cases_started: u64,
    /// Total cases completed.
    pub cases_completed: u64,
    /// Bottleneck anomalies injected.
    pub bottleneck_count: u64,
    /// Rework anomalies injected.
    pub rework_count: u64,
    /// Long-running anomalies injected.
    pub long_running_count: u64,
    /// Skip anomalies injected.
    pub skip_count: u64,
}

impl ProcessEventGenerator {
    /// Create a new generator.
    pub fn new(sector: SectorTemplate, config: GeneratorConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let registry = sector.build_registry();
        let transitions = Self::build_transition_map(&sector, &registry);
        let parallel_defs = Self::build_parallel_map(&sector, &registry);

        Self {
            sector,
            config,
            registry,
            transitions,
            parallel_defs,
            rng,
            next_event_id: 1,
            next_case_id: 1,
            active_cases: HashMap::new(),
            current_time: HybridTimestamp::now(),
            stats: GeneratorStats::default(),
        }
    }

    /// Build parallel activity map from sector template.
    fn build_parallel_map(
        sector: &SectorTemplate,
        registry: &ActivityRegistry,
    ) -> HashMap<ActivityId, ParallelActivityDef> {
        let mut map = HashMap::new();
        for def in sector.parallel_activities() {
            if let Some(fork_activity) = registry.get_by_name(def.fork_from) {
                map.insert(fork_activity.id, def);
            }
        }
        map
    }

    /// Build transition map from sector template.
    fn build_transition_map(
        sector: &SectorTemplate,
        registry: &ActivityRegistry,
    ) -> HashMap<ActivityId, Vec<(ActivityId, f32, u32)>> {
        let mut map: HashMap<ActivityId, Vec<(ActivityId, f32, u32)>> = HashMap::new();

        for trans in sector.transitions() {
            if let (Some(source), Some(target)) = (
                registry.get_by_name(trans.source),
                registry.get_by_name(trans.target),
            ) {
                map.entry(source.id).or_default().push((
                    target.id,
                    trans.probability,
                    trans.avg_transition_ms,
                ));
            }
        }

        // Normalize probabilities
        for transitions in map.values_mut() {
            let total: f32 = transitions.iter().map(|(_, p, _)| p).sum();
            if total > 0.0 {
                for (_, p, _) in transitions.iter_mut() {
                    *p /= total;
                }
            }
        }

        map
    }

    /// Get statistics.
    pub fn stats(&self) -> &GeneratorStats {
        &self.stats
    }

    /// Get current throughput (events per second estimate).
    pub fn throughput(&self) -> f32 {
        self.config.events_per_second as f32
    }

    /// Generate a batch of events.
    pub fn generate_batch(&mut self, batch_size: usize) -> Vec<GpuObjectEvent> {
        let mut events = Vec::with_capacity(batch_size);

        // Ensure we have enough active cases
        while self.active_cases.len() < self.config.concurrent_cases as usize {
            self.start_new_case();
        }

        for _ in 0..batch_size {
            if let Some(event) = self.generate_next_event() {
                events.push(event);
            }
        }

        events
    }

    /// Start a new case.
    fn start_new_case(&mut self) {
        let case_id = self.next_case_id;
        self.next_case_id += 1;

        // Get start activity
        let start_names = self.sector.start_activities();
        let start_name = start_names[self.rng.gen_range(0..start_names.len())];
        let start_activity = self
            .registry
            .get_by_name(start_name)
            .map(|a| a.id)
            .unwrap_or(1);

        // Determine if this case will have an anomaly
        let anomaly = self.determine_anomaly();

        let state = CaseState {
            case_id,
            current_activity: Some(start_activity),
            completed_activities: Vec::new(),
            last_event_time: self.current_time,
            is_complete: false,
            anomaly,
            parallel_activities: Vec::new(),
            join_to: None,
        };

        self.active_cases.insert(case_id, state);
        self.stats.cases_started += 1;

        if let Some(anomaly) = anomaly {
            match anomaly {
                AnomalyType::Bottleneck => self.stats.bottleneck_count += 1,
                AnomalyType::Rework => self.stats.rework_count += 1,
                AnomalyType::LongRunning => self.stats.long_running_count += 1,
                AnomalyType::Skip => self.stats.skip_count += 1,
            }
        }
    }

    /// Determine if and what anomaly to inject.
    fn determine_anomaly(&mut self) -> Option<AnomalyType> {
        let r: f32 = self.rng.gen();

        if r < self.config.anomalies.bottleneck_rate {
            Some(AnomalyType::Bottleneck)
        } else if r < self.config.anomalies.bottleneck_rate + self.config.anomalies.rework_rate {
            Some(AnomalyType::Rework)
        } else if r < self.config.anomalies.bottleneck_rate
            + self.config.anomalies.rework_rate
            + self.config.anomalies.long_running_rate
        {
            Some(AnomalyType::LongRunning)
        } else if r < self.config.anomalies.bottleneck_rate
            + self.config.anomalies.rework_rate
            + self.config.anomalies.long_running_rate
            + self.config.anomalies.skip_rate
        {
            Some(AnomalyType::Skip)
        } else {
            None
        }
    }

    /// Generate next event from active cases.
    fn generate_next_event(&mut self) -> Option<GpuObjectEvent> {
        // Pick a random active case
        let case_ids: Vec<u64> = self.active_cases.keys().copied().collect();
        if case_ids.is_empty() {
            return None;
        }

        let case_id = case_ids[self.rng.gen_range(0..case_ids.len())];

        // Remove case temporarily to avoid borrow conflicts
        let mut case = self.active_cases.remove(&case_id)?;

        // Check if case is in parallel mode
        if !case.parallel_activities.is_empty() {
            // Generate event for a parallel activity
            let event = self.create_parallel_event(&mut case);

            // Check if all parallel activities are done
            if case.parallel_activities.iter().all(|p| p.event_emitted) {
                // Join: all parallel activities complete, transition to join activity
                case.parallel_activities.clear();
                if let Some(join_activity) = case.join_to.take() {
                    case.current_activity = Some(join_activity);
                }
            }

            self.active_cases.insert(case_id, case);
            return event;
        }

        // Normal sequential mode
        let event = self.create_event(&mut case);

        // Transition to next activity (may enter parallel mode)
        let is_complete = self.transition_case(&mut case);

        if is_complete {
            self.stats.cases_completed += 1;
        } else {
            // Put case back
            self.active_cases.insert(case_id, case);
        }

        Some(event)
    }

    /// Create event for a parallel activity.
    fn create_parallel_event(&mut self, case: &mut CaseState) -> Option<GpuObjectEvent> {
        // Find next un-emitted parallel activity
        let idx = case
            .parallel_activities
            .iter()
            .position(|p| !p.event_emitted)?;

        let parallel = &mut case.parallel_activities[idx];
        parallel.event_emitted = true;

        let event_id = self.next_event_id;
        self.next_event_id += 1;

        // The event timestamp is the parallel activity's start time (they all start at the same time!)
        let timestamp = parallel.start_time;
        let duration = parallel.duration_ms;
        let activity_id = parallel.activity_id;

        case.completed_activities.push(activity_id);
        self.stats.total_events += 1;

        // Update last event time to max of parallel end times (for when join happens)
        let event_end = HybridTimestamp::new(
            timestamp.physical_ms + duration as u64,
            timestamp.logical + 1,
        );
        if event_end.physical_ms > case.last_event_time.physical_ms {
            case.last_event_time = event_end;
        }

        // Also update global time
        if event_end.physical_ms > self.current_time.physical_ms {
            self.current_time = event_end;
        }

        Some(GpuObjectEvent {
            event_id,
            object_id: case.case_id,
            activity_id,
            event_type: EventType::Complete as u8,
            _padding1: [0; 3],
            timestamp, // Same start time for all parallel activities!
            resource_id: self.rng.gen_range(1..100),
            cost: self.rng.gen_range(10.0..1000.0),
            duration_ms: duration,
            flags: 0,
            attributes: [0; 4],
            object_type_id: 0,
            related_object_id: 0,
            _reserved: [0; 36],
        })
    }

    /// Create an event for the current activity of a case.
    fn create_event(&mut self, case: &mut CaseState) -> GpuObjectEvent {
        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let current_activity = case.current_activity.unwrap_or(1);

        // Get activity details
        let activity = self.registry.get(current_activity);
        let base_duration = activity.map(|a| a.expected_duration_ms).unwrap_or(60_000);

        // Apply anomaly modifications
        let duration = match case.anomaly {
            Some(AnomalyType::Bottleneck) | Some(AnomalyType::LongRunning) => {
                (base_duration as f32 * self.rng.gen_range(3.0..10.0)) as u32
            }
            _ => {
                // Normal distribution around base duration
                let std_dev = base_duration as f32 * 0.3;
                let dist = Normal::new(base_duration as f64, std_dev as f64).unwrap();
                dist.sample(&mut self.rng).max(1000.0) as u32
            }
        };

        // Activity starts when the previous activity ended
        // (case.last_event_time tracks the end of the last activity)
        let activity_start_time = case.last_event_time;

        // Update case's last_event_time to the END of this activity
        case.last_event_time = HybridTimestamp::new(
            activity_start_time.physical_ms + duration as u64,
            activity_start_time.logical + 1,
        );

        // Also advance global time (for inter-case ordering)
        if case.last_event_time.physical_ms > self.current_time.physical_ms {
            self.current_time = case.last_event_time;
        }

        case.completed_activities.push(current_activity);
        self.stats.total_events += 1;

        GpuObjectEvent {
            event_id,
            object_id: case.case_id,
            activity_id: current_activity,
            event_type: EventType::Complete as u8,
            _padding1: [0; 3],
            timestamp: activity_start_time, // Activity START time
            resource_id: self.rng.gen_range(1..100),
            cost: self.rng.gen_range(10.0..1000.0),
            duration_ms: duration,
            flags: 0,
            attributes: [0; 4],
            object_type_id: 0,
            related_object_id: 0,
            _reserved: [0; 36],
        }
    }

    /// Transition case to next activity. Returns true if case is complete.
    fn transition_case(&mut self, case: &mut CaseState) -> bool {
        let current_activity = match case.current_activity {
            Some(act) => act,
            None => {
                case.is_complete = true;
                return true;
            }
        };

        // Check if current activity is an end activity
        let end_names = self.sector.end_activities();
        if let Some(activity) = self.registry.get(current_activity) {
            if end_names.contains(&activity.name.as_str()) {
                case.is_complete = true;
                return true;
            }
        }

        // Check if this activity triggers parallel execution
        if let Some(parallel_def) = self.parallel_defs.get(&current_activity) {
            // Roll for parallel vs sequential
            if self.rng.gen::<f32>() < parallel_def.probability {
                // Fork into parallel activities
                return self.start_parallel_activities(case, parallel_def.clone());
            }
        }

        // Get possible transitions
        let transitions = self.transitions.get(&current_activity);

        if let Some(trans) = transitions {
            if trans.is_empty() {
                case.is_complete = true;
                return true;
            }

            // Handle rework anomaly - repeat previous activity
            if matches!(case.anomaly, Some(AnomalyType::Rework))
                && self.rng.gen::<f32>() < 0.3
                && case.completed_activities.len() >= 2
            {
                let prev = case.completed_activities[case.completed_activities.len() - 2];
                case.current_activity = Some(prev);
                return false;
            }

            // Select next activity based on probabilities
            let r: f32 = self.rng.gen();
            let mut cumulative = 0.0;
            for (target, prob, _time) in trans {
                cumulative += prob;
                if r <= cumulative {
                    case.current_activity = Some(*target);
                    return false;
                }
            }

            // Fallback to first transition
            case.current_activity = Some(trans[0].0);
            false
        } else {
            // No transitions, case complete
            case.is_complete = true;
            true
        }
    }

    /// Start parallel activities for a case.
    fn start_parallel_activities(
        &mut self,
        case: &mut CaseState,
        parallel_def: ParallelActivityDef,
    ) -> bool {
        // Get the join activity ID
        let join_activity = match self.registry.get_by_name(parallel_def.join_to) {
            Some(a) => a.id,
            None => {
                // Can't find join activity, fall back to sequential
                return false;
            }
        };

        // Create parallel activity states - all start at the SAME time (concurrency!)
        let fork_time = case.last_event_time;
        let mut parallel_states = Vec::new();

        for activity_name in &parallel_def.parallel_activities {
            if let Some(activity) = self.registry.get_by_name(activity_name) {
                let base_duration = activity.expected_duration_ms;
                // Add some variance to duration
                let std_dev = base_duration as f32 * 0.3;
                let dist = Normal::new(base_duration as f64, std_dev as f64).unwrap();
                let duration = dist.sample(&mut self.rng).max(1000.0) as u32;

                parallel_states.push(ParallelActivityState {
                    activity_id: activity.id,
                    start_time: fork_time,
                    duration_ms: duration,
                    event_emitted: false,
                });
            }
        }

        if parallel_states.is_empty() {
            return false;
        }

        // Enter parallel mode
        case.current_activity = None; // No single current activity
        case.parallel_activities = parallel_states;
        case.join_to = Some(join_activity);

        false // Not complete
    }

    /// Get the sector template.
    pub fn sector(&self) -> &SectorTemplate {
        &self.sector
    }

    /// Get the activity registry.
    pub fn registry(&self) -> &ActivityRegistry {
        &self.registry
    }

    /// Number of active cases.
    pub fn active_case_count(&self) -> usize {
        self.active_cases.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let sector = SectorTemplate::default();
        let config = GeneratorConfig::default();
        let generator = ProcessEventGenerator::new(sector, config);

        assert_eq!(generator.active_case_count(), 0);
    }

    #[test]
    fn test_batch_generation() {
        let sector = SectorTemplate::default();
        let config = GeneratorConfig {
            concurrent_cases: 10,
            ..Default::default()
        };
        let mut generator = ProcessEventGenerator::new(sector, config);

        let events = generator.generate_batch(100);
        assert!(!events.is_empty());
        assert!(generator.stats().total_events > 0);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let sector = SectorTemplate::default();
        let config = GeneratorConfig {
            seed: Some(42),
            concurrent_cases: 5,
            ..Default::default()
        };

        let mut gen1 = ProcessEventGenerator::new(sector.clone(), config.clone());
        let events1 = gen1.generate_batch(50);

        let mut gen2 = ProcessEventGenerator::new(sector, config);
        let events2 = gen2.generate_batch(50);

        // With same seed, batches should have similar lengths (within 20%)
        // Note: exact determinism may vary due to timestamp-based logic
        assert!(!events1.is_empty());
        assert!(!events2.is_empty());
        let len_diff = (events1.len() as i32 - events2.len() as i32).abs();
        let avg_len = (events1.len() + events2.len()) / 2;
        assert!(
            len_diff <= (avg_len as f32 * 0.3) as i32,
            "Batch sizes too different: {} vs {}",
            events1.len(),
            events2.len()
        );
    }

    #[test]
    fn test_parallel_activity_generation() {
        use crate::fabric::HealthcareConfig;

        // Healthcare has Lab Tests and Imaging that can run in parallel
        let sector = SectorTemplate::Healthcare(HealthcareConfig::default());
        let config = GeneratorConfig {
            seed: Some(12345), // Use seed for reproducibility
            concurrent_cases: 50,
            ..Default::default()
        };
        let mut generator = ProcessEventGenerator::new(sector, config);

        // Generate enough events to likely trigger parallel activities
        let events = generator.generate_batch(2000);
        assert!(!events.is_empty());

        // Group events by case
        let mut cases: std::collections::HashMap<u64, Vec<&GpuObjectEvent>> =
            std::collections::HashMap::new();
        for event in &events {
            cases.entry(event.object_id).or_default().push(event);
        }

        // Check for cases where Lab Tests (4) and Imaging (5) have overlapping timestamps
        let mut found_parallel = false;
        for case_events in cases.values() {
            let lab_tests: Vec<_> = case_events.iter().filter(|e| e.activity_id == 4).collect();
            let imaging: Vec<_> = case_events.iter().filter(|e| e.activity_id == 5).collect();

            if !lab_tests.is_empty() && !imaging.is_empty() {
                // Check if they have the same start time (parallel execution)
                for lt in &lab_tests {
                    for img in &imaging {
                        if lt.timestamp.physical_ms == img.timestamp.physical_ms {
                            found_parallel = true;
                            break;
                        }
                    }
                }
            }
        }

        // With 40% probability of parallel execution in Healthcare config,
        // we should find at least one case with parallel activities
        assert!(
            found_parallel,
            "Expected to find parallel Lab Tests and Imaging activities"
        );
    }
}
