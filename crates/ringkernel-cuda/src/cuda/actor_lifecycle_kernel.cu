/**
 * RingKernel Actor Lifecycle Kernel
 *
 * Demonstrates the full GPU actor model:
 * - Block 0: Supervisor (manages actor lifecycle)
 * - Blocks 1-N: Actor pool (dormant until activated)
 * - H2K commands control create/destroy/restart
 * - K2H responses report lifecycle events and metrics
 * - Heartbeat monitoring for failure detection
 *
 * This kernel proves that GPU thread blocks can function as true actors
 * with dynamic lifecycle management, all within a single persistent launch.
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// ============================================================================
// Shared Structures (must match Rust definitions)
// ============================================================================

#define MAX_ACTORS 128
#define QUEUE_CAPACITY 64
#define QUEUE_MASK (QUEUE_CAPACITY - 1)

// Actor states (match Rust ActorState enum)
#define STATE_DORMANT      0
#define STATE_INITIALIZING 1
#define STATE_ACTIVE       2
#define STATE_DRAINING     3
#define STATE_TERMINATED   4
#define STATE_FAILED       5

// H2K commands (match Rust SimCommand enum)
#define CMD_NOP            0
#define CMD_TERMINATE      4
#define CMD_CREATE_ACTOR   16
#define CMD_DESTROY_ACTOR  17
#define CMD_RESTART_ACTOR  18
#define CMD_HEARTBEAT_REQ  19

// K2H response types (match Rust ResponseType enum)
#define RESP_ACK             0
#define RESP_PROGRESS        1
#define RESP_ERROR           2
#define RESP_TERMINATED      3
#define RESP_ACTOR_CREATED   16
#define RESP_ACTOR_DESTROYED 17
#define RESP_ACTOR_RESTARTED 18
#define RESP_HEARTBEAT       19
#define RESP_ACTOR_FAILED    20

// SPSC queue header (64 bytes, matches Rust SpscQueueHeader)
struct QueueHeader {
    unsigned long long head;
    unsigned long long tail;
    unsigned int capacity;
    unsigned int mask;
    unsigned long long _reserved[5];
};

// H2K message (64 bytes)
struct H2KMessage {
    unsigned int cmd;
    unsigned int flags;
    unsigned long long cmd_id;
    unsigned long long param1;  // actor_slot for lifecycle commands
    unsigned int param2;        // parent_slot for CreateActor
    unsigned int param3;
    float param4;
    float param5;
    unsigned long long _reserved[3];
};

// K2H message (64 bytes)
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

// Per-actor control block in mapped memory
struct ActorControl {
    unsigned int state;              // ActorState
    unsigned int parent_id;          // Parent actor block index
    unsigned int restart_count;      // Number of restarts
    unsigned int _pad0;
    unsigned long long heartbeat_ns; // Last heartbeat (nanoseconds)
    unsigned long long msgs_processed;
    unsigned long long total_work;   // Accumulated "work units" done
    unsigned int queue_depth;        // Current message count
    unsigned int _pad1;
    unsigned long long _reserved[2];
};

// ============================================================================
// Device Helpers
// ============================================================================

__device__ unsigned long long device_clock_ns() {
    unsigned long long clock;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock));
    return clock;
}

// Try to receive an H2K message (thread 0 only)
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

// Send a K2H response (thread 0 only)
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
// Actor Lifecycle Kernel
// ============================================================================

/**
 * Persistent kernel implementing the full actor lifecycle model.
 *
 * Block 0 = Supervisor:
 *   - Reads H2K commands for lifecycle operations
 *   - Manages actor state transitions
 *   - Monitors heartbeats
 *   - Sends K2H responses
 *
 * Blocks 1-N = Actor Pool:
 *   - Dormant blocks: spin on state flag (near-zero cost)
 *   - Active blocks: process "work" and update metrics
 *   - Respond to heartbeat requests
 */
extern "C" __global__ void actor_lifecycle_kernel(
    volatile QueueHeader* h2k_header,
    volatile H2KMessage* h2k_slots,
    volatile QueueHeader* k2h_header,
    volatile K2HMessage* k2h_slots,
    ActorControl* actor_controls,    // Array of MAX_ACTORS entries
    unsigned int* terminate_flag,    // Global termination flag
    unsigned int num_actors          // Number of actor slots (gridDim.x - 1)
) {
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    bool is_supervisor = (block_id == 0);

    // ─── Supervisor Block (Block 0) ─────────────────────────────────
    if (is_supervisor) {
        unsigned long long loop_count = 0;

        while (true) {
            // Check global termination
            if (atomicAdd(terminate_flag, 0) != 0) break;

            // Process H2K commands (thread 0 only)
            if (tid == 0) {
                H2KMessage cmd;
                while (h2k_try_recv(h2k_header, h2k_slots, &cmd)) {
                    K2HMessage resp;
                    resp.cmd_id = cmd.cmd_id;
                    resp.status = 0;

                    switch (cmd.cmd) {
                        case CMD_CREATE_ACTOR: {
                            unsigned int slot = (unsigned int)cmd.param1;
                            unsigned int parent = cmd.param2;

                            if (slot > 0 && slot <= num_actors) {
                                actor_controls[slot].state = STATE_ACTIVE;
                                actor_controls[slot].parent_id = parent;
                                actor_controls[slot].heartbeat_ns = device_clock_ns();
                                actor_controls[slot].msgs_processed = 0;
                                actor_controls[slot].total_work = 0;
                                actor_controls[slot].queue_depth = 0;

                                resp.resp_type = RESP_ACTOR_CREATED;
                                resp.param1 = slot;
                            } else {
                                resp.resp_type = RESP_ERROR;
                                resp.param1 = slot;
                            }
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                        }

                        case CMD_DESTROY_ACTOR: {
                            unsigned int slot = (unsigned int)cmd.param1;
                            if (slot > 0 && slot <= num_actors) {
                                actor_controls[slot].state = STATE_TERMINATED;
                                __threadfence();

                                // Wait briefly for actor to notice
                                for (int w = 0; w < 100; w++) {
                                    __nanosleep(100);
                                }
                                actor_controls[slot].state = STATE_DORMANT;

                                resp.resp_type = RESP_ACTOR_DESTROYED;
                                resp.param1 = slot;
                                resp.param2 = (unsigned int)actor_controls[slot].msgs_processed;
                            } else {
                                resp.resp_type = RESP_ERROR;
                                resp.param1 = slot;
                            }
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                        }

                        case CMD_RESTART_ACTOR: {
                            unsigned int slot = (unsigned int)cmd.param1;
                            if (slot > 0 && slot <= num_actors) {
                                // Save parent
                                unsigned int parent = actor_controls[slot].parent_id;
                                unsigned int restarts = actor_controls[slot].restart_count;

                                // Destroy
                                actor_controls[slot].state = STATE_TERMINATED;
                                __threadfence();
                                for (int w = 0; w < 50; w++) __nanosleep(100);

                                // Re-create with incremented restart count
                                actor_controls[slot].state = STATE_ACTIVE;
                                actor_controls[slot].parent_id = parent;
                                actor_controls[slot].restart_count = restarts + 1;
                                actor_controls[slot].heartbeat_ns = device_clock_ns();
                                actor_controls[slot].msgs_processed = 0;
                                actor_controls[slot].total_work = 0;

                                resp.resp_type = RESP_ACTOR_RESTARTED;
                                resp.param1 = slot;
                                resp.param2 = restarts + 1;
                            } else {
                                resp.resp_type = RESP_ERROR;
                                resp.param1 = slot;
                            }
                            k2h_send(k2h_header, k2h_slots, &resp);
                            break;
                        }

                        case CMD_HEARTBEAT_REQ: {
                            unsigned int slot = (unsigned int)cmd.param1;
                            if (slot > 0 && slot <= num_actors) {
                                resp.resp_type = RESP_HEARTBEAT;
                                resp.param1 = slot;
                                resp.param2 = (unsigned int)actor_controls[slot].msgs_processed;
                                resp.param4 = (float)(device_clock_ns() - actor_controls[slot].heartbeat_ns) / 1e6f; // ms since last
                            } else {
                                resp.resp_type = RESP_ERROR;
                                resp.param1 = slot;
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
            }

            __syncthreads();

            loop_count++;
            if (loop_count % 1000 == 0) {
                // Periodic: check for heartbeat timeouts
                // (simplified: supervisor just updates its own heartbeat)
                if (tid == 0) {
                    actor_controls[0].heartbeat_ns = device_clock_ns();
                    actor_controls[0].msgs_processed = loop_count;
                }
            }

            grid.sync();
        }

        return;
    }

    // ─── Actor Blocks (Block 1-N) ───────────────────────────────────
    unsigned int my_slot = block_id;
    if (my_slot > num_actors) return;

    volatile ActorControl* my_ctrl = &actor_controls[my_slot];
    unsigned long long work_accumulator = 0;

    while (true) {
        // Check global termination
        if (atomicAdd(terminate_flag, 0) != 0) break;

        // Check my state
        unsigned int state = my_ctrl->state;

        if (state == STATE_ACTIVE) {
            // === Active: do "work" ===
            // Simulate message processing: each thread computes something
            float val = (float)(tid + my_slot * 256);
            for (int i = 0; i < 10; i++) {
                val = val * 0.999f + 0.001f;  // Trivial work
            }

            // Thread 0 updates metrics
            if (tid == 0) {
                my_ctrl->msgs_processed++;
                work_accumulator++;
                my_ctrl->total_work = work_accumulator;
                my_ctrl->heartbeat_ns = device_clock_ns();
            }

        } else if (state == STATE_TERMINATED || state == STATE_DRAINING) {
            // === Terminating: clean up and go dormant ===
            if (tid == 0) {
                my_ctrl->total_work = work_accumulator;
            }
            // Wait for supervisor to set us to DORMANT
            // (we stay in this branch until state changes)
        }
        // STATE_DORMANT: just loop (near-zero cost - the branch is cheap)

        grid.sync();
    }
}
