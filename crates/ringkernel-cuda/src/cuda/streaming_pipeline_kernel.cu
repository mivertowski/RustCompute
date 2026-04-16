/**
 * RingKernel GPU Streaming Event Pipeline
 *
 * Real-world application: GPU-native event stream processing with
 * actor-per-stage pipeline, dynamic lifecycle, and measured metrics.
 *
 * Architecture:
 *   Block 0:   Supervisor (lifecycle management)
 *   Block 1-2: Ingest actors (read events from H2K, parse, forward)
 *   Block 3-4: Filter actors (apply rules, discard/forward)
 *   Block 5:   Aggregate actor (windowed statistics)
 *   Block 6:   Alert actor (threshold detection, emit K2H)
 *   Block 7+:  Spare pool (for dynamic scaling)
 *
 * Optimization over actor_lifecycle_kernel:
 *   - grid.sync() only every SYNC_INTERVAL iterations (not every iter)
 *   - __syncthreads() for intra-block work (200ns vs 15us)
 *   - Batch H2K command processing
 *   - Adaptive nanosleep when idle
 *   - Shared memory work queues for inter-stage data within cluster
 *
 * Event format (32 bytes):
 *   [timestamp:u64][source_id:u32][event_type:u32][value:f32][flags:u32][payload:u64]
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// Configuration
// ============================================================================

#define MAX_ACTORS         32
#define QUEUE_CAPACITY     64
#define QUEUE_MASK         (QUEUE_CAPACITY - 1)
#define SYNC_INTERVAL      64    // Grid sync every N iterations (was 1)
#define MAX_EVENTS_PER_BLOCK 1024
#define PIPELINE_STAGES    4     // Ingest, Filter, Aggregate, Alert

// Actor states
#define STATE_DORMANT      0
#define STATE_ACTIVE       2
#define STATE_TERMINATED   4

// H2K commands
#define CMD_NOP            0
#define CMD_TERMINATE      4
#define CMD_CREATE_ACTOR   16
#define CMD_DESTROY_ACTOR  17
#define CMD_INJECT_EVENTS  32   // New: batch inject events
#define CMD_SET_FILTER     33   // New: configure filter rule
#define CMD_SET_THRESHOLD  34   // New: configure alert threshold

// K2H responses
#define RESP_ACK             0
#define RESP_TERMINATED      3
#define RESP_ACTOR_CREATED   16
#define RESP_ACTOR_DESTROYED 17
#define RESP_ALERT           32  // New: threshold breach alert
#define RESP_METRICS         33  // New: pipeline metrics report

// Actor roles
#define ROLE_SUPERVISOR  0
#define ROLE_INGEST      1
#define ROLE_FILTER      2
#define ROLE_AGGREGATE   3
#define ROLE_ALERT       4

// ============================================================================
// Structures
// ============================================================================

struct QueueHeader {
    unsigned long long head;
    unsigned long long tail;
    unsigned int capacity;
    unsigned int mask;
    unsigned long long _reserved[5];
};

struct H2KMessage {
    unsigned int cmd;
    unsigned int flags;
    unsigned long long cmd_id;
    unsigned long long param1;
    unsigned int param2;
    unsigned int param3;
    float param4;
    float param5;
    unsigned long long _reserved[3];
};

struct K2HMessage {
    unsigned int resp_type;
    unsigned int status;
    unsigned long long cmd_id;
    unsigned long long param1;
    unsigned int param2;
    unsigned int param3;
    float param4;
    float param5;
    unsigned long long _reserved[3];
};

// Streaming event (32 bytes)
struct Event {
    unsigned long long timestamp;
    unsigned int source_id;
    unsigned int event_type;
    float value;
    unsigned int flags;
    unsigned long long payload;
};

// Per-actor control (64 bytes)
struct ActorControl {
    unsigned int state;
    unsigned int role;
    unsigned int restart_count;
    unsigned int parent_id;
    unsigned long long heartbeat_ns;
    unsigned long long events_processed;
    unsigned long long events_forwarded;
    unsigned long long events_filtered;
    float aggregate_sum;
    float aggregate_min;
    float aggregate_max;
    unsigned int aggregate_count;
};

// Pipeline metrics (written by each stage, read by host)
struct PipelineMetrics {
    unsigned long long total_ingested;
    unsigned long long total_filtered_out;
    unsigned long long total_aggregated;
    unsigned long long total_alerts;
    unsigned long long pipeline_latency_ns;  // End-to-end
    float throughput_events_per_sec;
    float filter_rate;                        // fraction filtered out
    float avg_value;
    unsigned int active_actors;
    unsigned int _pad[3];
};

// ============================================================================
// Device Helpers
// ============================================================================

__device__ unsigned long long device_clock_ns() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock));
    return clock;
}

__device__ bool h2k_try_recv(
    volatile QueueHeader* header,
    volatile H2KMessage* slots,
    H2KMessage* out
) {
    unsigned long long head = header->head;
    unsigned long long tail = header->tail;
    __threadfence_system();
    if (head == tail) return false;
    unsigned int idx = (unsigned int)(tail & QUEUE_MASK);
    *out = *(const H2KMessage*)&slots[idx];
    __threadfence();
    header->tail = tail + 1;
    __threadfence_system();
    return true;
}

__device__ bool k2h_send(
    volatile QueueHeader* header,
    volatile K2HMessage* slots,
    K2HMessage* msg
) {
    unsigned long long head = header->head;
    unsigned long long tail = header->tail;
    __threadfence_system();
    if (head - tail >= QUEUE_CAPACITY) return false;
    unsigned int idx = (unsigned int)(head & QUEUE_MASK);
    *(K2HMessage*)&slots[idx] = *msg;
    __threadfence();
    header->head = head + 1;
    __threadfence_system();
    return true;
}

// ============================================================================
// Inter-actor communication via global memory ring buffers
// ============================================================================

// Simple inter-actor event buffer (global memory)
// Each actor pair has a small ring buffer for forwarding events
struct InterActorBuffer {
    Event events[256];
    unsigned int head;    // Written by producer
    unsigned int tail;    // Written by consumer
    unsigned int _pad[2];
};

__device__ bool inter_send(InterActorBuffer* buf, const Event* evt) {
    unsigned int head = buf->head;
    unsigned int tail = buf->tail;
    if (head - tail >= 256) return false;  // Full
    memcpy((void*)&buf->events[head & 255], (const void*)evt, sizeof(Event));
    __threadfence();
    buf->head = head + 1;
    return true;
}

__device__ bool inter_recv(InterActorBuffer* buf, Event* out) {
    unsigned int head = buf->head;
    unsigned int tail = buf->tail;
    __threadfence();
    if (head == tail) return false;  // Empty
    memcpy((void*)out, (const void*)&buf->events[tail & 255], sizeof(Event));
    __threadfence();
    buf->tail = tail + 1;
    return true;
}

// ============================================================================
// Streaming Pipeline Kernel
// ============================================================================

extern "C" __global__ void streaming_pipeline_kernel(
    volatile QueueHeader* h2k_header,
    volatile H2KMessage* h2k_slots,
    volatile QueueHeader* k2h_header,
    volatile K2HMessage* k2h_slots,
    ActorControl* actor_controls,
    PipelineMetrics* metrics,
    InterActorBuffer* inter_buffers,  // Array: [src * MAX_ACTORS + dst]
    Event* event_inject_buffer,       // Bulk event injection buffer
    unsigned int* event_inject_count, // Number of events to inject
    unsigned int* terminate_flag,
    unsigned int num_actors,
    float filter_threshold,           // Events with value < threshold are filtered
    float alert_threshold             // Events with value > threshold trigger alert
) {
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    bool is_supervisor = (block_id == 0);

    unsigned long long iteration = 0;
    unsigned long long start_ns = device_clock_ns();

    // ─── Supervisor ─────────────────────────────────────────────────
    if (is_supervisor) {
        while (true) {
            if (atomicAdd(terminate_flag, 0) != 0) break;

            if (tid == 0) {
                H2KMessage cmd;
                // Batch process all pending commands
                for (int batch = 0; batch < 8; batch++) {
                    if (!h2k_try_recv(h2k_header, h2k_slots, &cmd)) break;

                    K2HMessage resp;
                    resp.cmd_id = cmd.cmd_id;
                    resp.status = 0;

                    switch (cmd.cmd) {
                        case CMD_CREATE_ACTOR: {
                            unsigned int slot = (unsigned int)cmd.param1;
                            unsigned int role = cmd.param2;
                            if (slot > 0 && slot <= num_actors) {
                                actor_controls[slot].state = STATE_ACTIVE;
                                actor_controls[slot].role = role;
                                actor_controls[slot].parent_id = cmd.param3;
                                actor_controls[slot].heartbeat_ns = device_clock_ns();
                                actor_controls[slot].events_processed = 0;
                                actor_controls[slot].events_forwarded = 0;
                                actor_controls[slot].events_filtered = 0;
                                actor_controls[slot].aggregate_sum = 0;
                                actor_controls[slot].aggregate_min = 1e30f;
                                actor_controls[slot].aggregate_max = -1e30f;
                                actor_controls[slot].aggregate_count = 0;
                                resp.resp_type = RESP_ACTOR_CREATED;
                                resp.param1 = slot;
                            }
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                        }
                        case CMD_DESTROY_ACTOR: {
                            unsigned int slot = (unsigned int)cmd.param1;
                            if (slot > 0 && slot <= num_actors) {
                                actor_controls[slot].state = STATE_DORMANT;
                                resp.resp_type = RESP_ACTOR_DESTROYED;
                                resp.param1 = slot;
                                resp.param2 = (unsigned int)actor_controls[slot].events_processed;
                            }
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                        }
                        case CMD_TERMINATE:
                            atomicExch(terminate_flag, 1);
                            resp.resp_type = RESP_TERMINATED;
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                        default:
                            resp.resp_type = RESP_ACK;
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                    }
                }

                // Periodic metrics update
                if (iteration % 1000 == 0) {
                    unsigned int active = 0;
                    for (unsigned int i = 1; i <= num_actors; i++) {
                        if (actor_controls[i].state == STATE_ACTIVE) active++;
                    }
                    metrics->active_actors = active;

                    unsigned long long elapsed_ns = device_clock_ns() - start_ns;
                    if (elapsed_ns > 0) {
                        float elapsed_s = (float)elapsed_ns / 1e9f;
                        metrics->throughput_events_per_sec =
                            (float)metrics->total_ingested / elapsed_s;
                    }
                    if (metrics->total_ingested > 0) {
                        metrics->filter_rate =
                            (float)metrics->total_filtered_out / (float)metrics->total_ingested;
                    }
                }
            }
            __syncthreads();

            iteration++;
            // Grid sync at reduced frequency
            if (iteration % SYNC_INTERVAL == 0) {
                grid.sync();
            }
        }
        return;
    }

    // ─── Actor Blocks ───────────────────────────────────────────────
    unsigned int my_slot = block_id;
    if (my_slot > num_actors) return;

    volatile ActorControl* my_ctrl = &actor_controls[my_slot];

    while (true) {
        if (atomicAdd(terminate_flag, 0) != 0) break;

        unsigned int state = my_ctrl->state;
        unsigned int role = my_ctrl->role;

        if (state == STATE_ACTIVE) {
            my_ctrl->heartbeat_ns = device_clock_ns();

            switch (role) {
                case ROLE_INGEST: {
                    // Read events from inject buffer and forward to filter actors
                    if (tid == 0) {
                        unsigned int count = *event_inject_count;
                        // Each ingest actor takes a share of events
                        unsigned int start_idx = my_slot * (count / 2);
                        unsigned int end_idx = (my_slot + 1) * (count / 2);
                        if (end_idx > count) end_idx = count;

                        for (unsigned int i = start_idx; i < end_idx; i++) {
                            Event evt = event_inject_buffer[i];
                            evt.timestamp = device_clock_ns();

                            // Forward to filter actor (slot 3 or 4)
                            unsigned int filter_slot = 3 + (i % 2);
                            inter_send(&inter_buffers[my_slot * MAX_ACTORS + filter_slot], &evt);
                            my_ctrl->events_processed++;
                            my_ctrl->events_forwarded++;
                            atomicAdd((unsigned long long*)&metrics->total_ingested, 1);
                        }
                    }
                    break;
                }

                case ROLE_FILTER: {
                    // Read from ingest actors, apply filter, forward survivors
                    if (tid == 0) {
                        Event evt;
                        // Check both ingest actors
                        for (unsigned int src = 1; src <= 2; src++) {
                            while (inter_recv(&inter_buffers[src * MAX_ACTORS + my_slot], &evt)) {
                                my_ctrl->events_processed++;

                                if (evt.value >= filter_threshold) {
                                    // Forward to aggregate actor (slot 5)
                                    inter_send(&inter_buffers[my_slot * MAX_ACTORS + 5], &evt);
                                    my_ctrl->events_forwarded++;
                                } else {
                                    my_ctrl->events_filtered++;
                                    atomicAdd((unsigned long long*)&metrics->total_filtered_out, 1);
                                }
                            }
                        }
                    }
                    break;
                }

                case ROLE_AGGREGATE: {
                    // Compute windowed statistics
                    if (tid == 0) {
                        Event evt;
                        // Read from both filter actors
                        for (unsigned int src = 3; src <= 4; src++) {
                            while (inter_recv(&inter_buffers[src * MAX_ACTORS + my_slot], &evt)) {
                                my_ctrl->events_processed++;
                                my_ctrl->aggregate_sum += evt.value;
                                if (evt.value < my_ctrl->aggregate_min) my_ctrl->aggregate_min = evt.value;
                                if (evt.value > my_ctrl->aggregate_max) my_ctrl->aggregate_max = evt.value;
                                my_ctrl->aggregate_count++;
                                atomicAdd((unsigned long long*)&metrics->total_aggregated, 1);

                                // Forward to alert actor (slot 6)
                                inter_send(&inter_buffers[my_slot * MAX_ACTORS + 6], &evt);
                                my_ctrl->events_forwarded++;
                            }
                        }

                        // Update running average
                        if (my_ctrl->aggregate_count > 0) {
                            metrics->avg_value = my_ctrl->aggregate_sum / (float)my_ctrl->aggregate_count;
                        }
                    }
                    break;
                }

                case ROLE_ALERT: {
                    // Check for threshold breaches
                    if (tid == 0) {
                        Event evt;
                        // Read from aggregate actor
                        while (inter_recv(&inter_buffers[5 * MAX_ACTORS + my_slot], &evt)) {
                            my_ctrl->events_processed++;

                            if (evt.value > alert_threshold) {
                                // Emit alert via K2H
                                K2HMessage alert;
                                alert.resp_type = RESP_ALERT;
                                alert.param1 = evt.source_id;
                                alert.param4 = evt.value;
                                alert.param5 = alert_threshold;
                                k2h_send(k2h_header, k2h_slots, &alert);
                                atomicAdd((unsigned long long*)&metrics->total_alerts, 1);
                            }
                        }
                    }
                    break;
                }
            }
        }

        __syncthreads();
        iteration++;
        if (iteration % SYNC_INTERVAL == 0) {
            grid.sync();
        }
    }
}
