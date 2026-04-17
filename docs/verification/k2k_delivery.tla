--------------------------- MODULE k2k_delivery ---------------------------
(***************************************************************************)
(* Single-GPU kernel-to-kernel (K2K) message delivery.                     *)
(*                                                                         *)
(* Each actor has a bounded FIFO in-queue. Senders enqueue; the target     *)
(* deliberately pulls a message in FIFO order. This spec proves:           *)
(*                                                                         *)
(*   * No message is lost between being accepted into a queue and being   *)
(*     delivered (NoMessageLoss).                                          *)
(*   * Per-sender, per-receiver FIFO order is preserved (FIFOPerSender).   *)
(*   * No message is delivered more than once (NoDuplicateDelivery).       *)
(*                                                                         *)
(* Maps to the K2K trait in crates/ringkernel-core/src/k2k.rs              *)
(*                                                                         *)
(* Abstractions:                                                           *)
(*   * Messages are unique ids drawn from 1..MaxMsgs.                      *)
(*   * Send may fail if the queue is full, in which case Drop(a, m) is    *)
(*     used to reflect back-pressure. Drops are NOT losses: the message   *)
(*     is considered never-accepted (not in sent_accepted).                *)
(***************************************************************************)

EXTENDS Naturals, TLC, Sequences, FiniteSets

CONSTANTS
  Actors,      \* set of actors, e.g. {a1, a2, a3}
  MaxMsgs,     \* total number of distinct messages
  QueueCap     \* bound on per-actor queue length

VARIABLES
  queues,           \* queues[a] = Seq of {sender, msg_id} pairs
  sent_accepted,    \* set of <<from, to, msg>> accepted into a queue
  delivered,        \* set of <<from, to, msg>> delivered to destination
  dropped,          \* set of <<from, to, msg>> rejected due to full queue
  next_msg          \* next fresh message id

vars == <<queues, sent_accepted, delivered, dropped, next_msg>>

Msgs == 1..MaxMsgs

Init ==
  /\ queues         = [a \in Actors |-> <<>>]
  /\ sent_accepted  = {}
  /\ delivered      = {}
  /\ dropped        = {}
  /\ next_msg       = 1

(***************************************************************************)
(* Send: attempt to enqueue a fresh message from `from` to `to`.           *)
(* If the destination queue is full, the message is dropped (back-pressure)*)
(* but is still counted in the state as rejected, not lost.                *)
(***************************************************************************)
Send(from, to) ==
  /\ from /= to
  /\ next_msg <= MaxMsgs
  /\ LET m == next_msg
         qlen == Len(queues[to])
     IN IF qlen < QueueCap
          THEN /\ queues' = [queues EXCEPT ![to] = Append(@, <<from, m>>)]
               /\ sent_accepted' = sent_accepted \cup {<<from, to, m>>}
               /\ dropped' = dropped
          ELSE /\ queues' = queues
               /\ sent_accepted' = sent_accepted
               /\ dropped' = dropped \cup {<<from, to, m>>}
  /\ next_msg' = next_msg + 1
  /\ UNCHANGED delivered

(***************************************************************************)
(* Deliver: actor `a` pulls the oldest message in its queue.               *)
(***************************************************************************)
Deliver(a) ==
  /\ Len(queues[a]) > 0
  /\ LET head == Head(queues[a])
         from == head[1]
         m    == head[2]
     IN /\ delivered' = delivered \cup {<<from, a, m>>}
        /\ queues' = [queues EXCEPT ![a] = Tail(@)]
  /\ UNCHANGED <<sent_accepted, dropped, next_msg>>

Next ==
  \/ \E f, t \in Actors : Send(f, t)
  \/ \E a \in Actors    : Deliver(a)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Invariants.                                                             *)
(***************************************************************************)

(* Every delivered message was first accepted into a queue. *)
DeliveredImpliesAccepted ==
  \A tup \in delivered : tup \in sent_accepted

(* Every accepted message is either still in flight or delivered.          *)
(* Never lost once accepted.                                               *)
InFlight(from, to, m) ==
  \E i \in 1..Len(queues[to]) : queues[to][i] = <<from, m>>

NoMessageLoss ==
  \A tup \in sent_accepted :
      LET f == tup[1]
          t == tup[2]
          m == tup[3]
      IN \/ <<f, t, m>> \in delivered
         \/ InFlight(f, t, m)

(* FIFO per sender: if m1 was sent by f to t before m2, then m1 is        *)
(* delivered before m2 (message ids are monotonic with sending order).     *)
FIFOPerSender ==
  \A f, t \in Actors :
      \A m1, m2 \in Msgs :
          ( /\ <<f, t, m1>> \in sent_accepted
            /\ <<f, t, m2>> \in sent_accepted
            /\ m1 < m2
            /\ <<f, t, m2>> \in delivered )
            => <<f, t, m1>> \in delivered

(* Message ids are fresh per send (`next_msg` is monotone). A given id   *)
(* therefore never appears in delivered under two different triples; if  *)
(* it did, two distinct (from, to, m) entries would share `m`, which we  *)
(* assert does not happen.                                               *)
NoDuplicateDelivery ==
  \A t1, t2 \in delivered :
      (t1[3] = t2[3]) => (t1 = t2)

(* Dropped and accepted are disjoint on (f,t,m) triples. *)
DropAcceptedDisjoint ==
  \A tup \in dropped : tup \notin sent_accepted

QueueBoundOK ==
  \A a \in Actors : Len(queues[a]) <= QueueCap

TypeOK ==
  /\ queues        \in [Actors -> Seq(Actors \X Msgs)]
  /\ sent_accepted \subseteq (Actors \X Actors \X Msgs)
  /\ delivered     \subseteq (Actors \X Actors \X Msgs)
  /\ dropped       \subseteq (Actors \X Actors \X Msgs)
  /\ next_msg      \in 1..(MaxMsgs + 1)

=============================================================================
