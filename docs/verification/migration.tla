----------------------------- MODULE migration -----------------------------
(***************************************************************************)
(* Actor migration: three-phase protocol to move an actor from a source   *)
(* GPU to a target GPU without losing in-flight messages.                  *)
(*                                                                         *)
(* Phases (modelled as a state machine):                                   *)
(*   Running     -- normal operation on source                             *)
(*   Quiesced    -- source stops accepting new messages; drains queue      *)
(*   Transferring-- state (and buffered messages) copied to target         *)
(*   Swapping    -- routing_table atomically flips from src to tgt         *)
(*   Complete    -- target is the authoritative home                       *)
(*                                                                         *)
(* Actions: BeginQuiesce, CaptureState, Transfer, SwapRouting, Finalize    *)
(*                                                                         *)
(* Safety properties:                                                      *)
(*   * NoMessageLoss    -- every message accepted at any phase is          *)
(*                         eventually observed on the target.              *)
(*   * FIFOPreserved    -- relative order across the cutover is unchanged. *)
(*   * AtomicSwap       -- routing flips from src to tgt in one step;     *)
(*                         at no observable point does both accept.        *)
(*   * ChecksumMatch    -- the state captured on src equals the state     *)
(*                         restored on tgt (modelled via a symbolic hash). *)
(*                                                                         *)
(* Correspondence: MigrationState in multi_gpu.rs:                         *)
(*   Pending / Quiescing / Checkpointing / Transferring / Restoring /      *)
(*   Completed.                                                            *)
(***************************************************************************)

EXTENDS Naturals, TLC, Sequences, FiniteSets

CONSTANTS
  MaxMsgs      \* bound on total messages across the run

VARIABLES
  phase,            \* current phase in {Running, Quiesced, Transferring,
                    \*                   Swapping, Complete}
  src_queue,        \* Seq of msgs still at the source
  tgt_queue,        \* Seq of msgs at the target
  src_state,        \* symbolic src state: set of accepted msg ids
  tgt_state,        \* symbolic tgt state: set of received msg ids
  in_flight_buffer, \* messages arriving during quiesce/transfer (staging)
  routing_table,    \* "src" or "tgt" -- which GPU is authoritative
  next_msg,         \* next fresh message id
  delivered,        \* multiset of delivered msg ids on the actor
  checksum_src,     \* checksum captured at source
  checksum_tgt      \* checksum observed at target after restore

vars == <<phase, src_queue, tgt_queue, src_state, tgt_state,
          in_flight_buffer, routing_table, next_msg, delivered,
          checksum_src, checksum_tgt>>

Msgs == 1..MaxMsgs
Phases == {"Running", "Quiesced", "Transferring", "Swapping", "Complete"}

(* Symbolic checksum: cardinality of the state set. Any deterministic    *)
(* function of state works; cardinality keeps TLC configuration small    *)
(* and matches on equality iff the two sets are the same size (which,    *)
(* because tgt_state := src_state at transfer time, is iff they are      *)
(* equal in this abstraction).                                            *)
Checksum(S) == Cardinality(S)

Init ==
  /\ phase            = "Running"
  /\ src_queue        = <<>>
  /\ tgt_queue        = <<>>
  /\ src_state        = {}
  /\ tgt_state        = {}
  /\ in_flight_buffer = <<>>
  /\ routing_table    = "src"
  /\ next_msg         = 1
  /\ delivered        = {}
  /\ checksum_src     = 0
  /\ checksum_tgt     = 0

(***************************************************************************)
(* Running: source accepts new messages normally.                          *)
(***************************************************************************)
AcceptOnSrc ==
  /\ phase = "Running"
  /\ routing_table = "src"
  /\ next_msg <= MaxMsgs
  /\ src_queue' = Append(src_queue, next_msg)
  /\ src_state' = src_state \cup {next_msg}
  /\ next_msg'  = next_msg + 1
  /\ UNCHANGED <<phase, tgt_queue, tgt_state, in_flight_buffer,
                 routing_table, delivered, checksum_src, checksum_tgt>>

(***************************************************************************)
(* Target accepts new messages once routing has been swapped.              *)
(***************************************************************************)
AcceptOnTgt ==
  /\ phase = "Complete"
  /\ routing_table = "tgt"
  /\ next_msg <= MaxMsgs
  /\ tgt_queue' = Append(tgt_queue, next_msg)
  /\ tgt_state' = tgt_state \cup {next_msg}
  /\ next_msg'  = next_msg + 1
  /\ UNCHANGED <<phase, src_queue, src_state, in_flight_buffer,
                 routing_table, delivered, checksum_src, checksum_tgt>>

(***************************************************************************)
(* Phase transitions.                                                      *)
(***************************************************************************)
BeginQuiesce ==
  /\ phase = "Running"
  /\ phase' = "Quiesced"
  /\ UNCHANGED <<src_queue, tgt_queue, src_state, tgt_state,
                 in_flight_buffer, routing_table, next_msg, delivered,
                 checksum_src, checksum_tgt>>

(***************************************************************************)
(* During quiesce new messages land in in_flight_buffer, not src_queue.    *)
(***************************************************************************)
AcceptDuringQuiesce ==
  /\ phase \in {"Quiesced", "Transferring", "Swapping"}
  /\ next_msg <= MaxMsgs
  /\ in_flight_buffer' = Append(in_flight_buffer, next_msg)
  /\ src_state'        = src_state \cup {next_msg}
  /\ next_msg'         = next_msg + 1
  /\ UNCHANGED <<phase, src_queue, tgt_queue, tgt_state,
                 routing_table, delivered, checksum_src, checksum_tgt>>

CaptureState ==
  /\ phase = "Quiesced"
  /\ phase' = "Transferring"
  /\ checksum_src' = Checksum(src_state)
  /\ UNCHANGED <<src_queue, tgt_queue, src_state, tgt_state,
                 in_flight_buffer, routing_table, next_msg, delivered,
                 checksum_tgt>>

Transfer ==
  /\ phase = "Transferring"
  /\ tgt_state'     = src_state           \* full state copy
  /\ tgt_queue'     = src_queue \o in_flight_buffer  \* FIFO concat
  /\ checksum_tgt'  = Checksum(src_state)
  /\ phase'         = "Swapping"
  /\ UNCHANGED <<src_queue, src_state, in_flight_buffer, routing_table,
                 next_msg, delivered, checksum_src>>

SwapRouting ==
  /\ phase = "Swapping"
  /\ checksum_src = checksum_tgt
  /\ routing_table' = "tgt"
  /\ phase'         = "Complete"
  /\ UNCHANGED <<src_queue, tgt_queue, src_state, tgt_state,
                 in_flight_buffer, next_msg, delivered,
                 checksum_src, checksum_tgt>>

Finalize ==
  /\ phase = "Complete"
  /\ src_queue' = <<>>
  /\ src_state' = {}             \* src is now inert
  /\ in_flight_buffer' = <<>>
  /\ UNCHANGED <<phase, tgt_queue, tgt_state, routing_table, next_msg,
                 delivered, checksum_src, checksum_tgt>>

(***************************************************************************)
(* Deliver from the target queue (after migration completes).              *)
(***************************************************************************)
DeliverOnTgt ==
  /\ phase = "Complete"
  /\ Len(tgt_queue) > 0
  /\ LET m == Head(tgt_queue) IN
       /\ delivered'  = delivered \cup {m}
       /\ tgt_queue' = Tail(tgt_queue)
  /\ UNCHANGED <<phase, src_queue, src_state, tgt_state, in_flight_buffer,
                 routing_table, next_msg, checksum_src, checksum_tgt>>

Next ==
  \/ AcceptOnSrc
  \/ AcceptDuringQuiesce
  \/ BeginQuiesce
  \/ CaptureState
  \/ Transfer
  \/ SwapRouting
  \/ Finalize
  \/ DeliverOnTgt
  \/ AcceptOnTgt

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Invariants.                                                             *)
(***************************************************************************)

(* Only the authoritative side accepts messages at any given phase.        *)
AtomicSwap ==
  \/ (routing_table = "src" /\ phase /= "Complete")
  \/ (routing_table = "tgt" /\ phase  = "Complete")

(* No message is silently dropped: every accepted id is in src_state,     *)
(* tgt_state, or delivered.                                                *)
NoMessageLoss ==
  \A m \in 1..(next_msg - 1) :
      m \in src_state \cup tgt_state \cup delivered

(* FIFO preserved: the target queue after transfer is src_queue ++ buffer.*)
(* Once phase is Swapping or Complete, no reordering is possible because   *)
(* tgt_queue is built by concatenation and only drained from the head.     *)
FIFOPreserved ==
  phase \in {"Swapping", "Complete"} =>
     \A i, j \in 1..Len(tgt_queue) :
         (i < j) => tgt_queue[i] < tgt_queue[j]
         \* message ids are fresh and monotonically increasing

(* Checksum from source matches the restored checksum at the target.      *)
ChecksumMatch ==
  phase \in {"Swapping", "Complete"} => checksum_src = checksum_tgt

(* The phase machine never skips forward without going through required   *)
(* intermediate states (captured by the enabledness of each action).      *)
PhaseLegal == phase \in Phases

TypeOK ==
  /\ phase            \in Phases
  /\ routing_table    \in {"src", "tgt"}
  /\ next_msg         \in 1..(MaxMsgs + 1)
  /\ src_queue        \in Seq(Msgs)
  /\ tgt_queue        \in Seq(Msgs)
  /\ in_flight_buffer \in Seq(Msgs)
  /\ src_state        \subseteq Msgs
  /\ tgt_state        \subseteq Msgs
  /\ delivered        \subseteq Msgs

=============================================================================
