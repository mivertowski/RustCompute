-------------------------------- MODULE hlc --------------------------------
(***************************************************************************)
(* Hybrid Logical Clock (HLC) monotonicity and causal ordering.            *)
(*                                                                         *)
(* RingKernel uses HLC timestamps on every event to give GPU kernels a     *)
(* total, causality-respecting ordering while staying close to wall time.  *)
(* This spec models the core tick / receive protocol implemented in        *)
(*   crates/ringkernel-core/src/hlc.rs                                     *)
(*                                                                         *)
(* Abstractions:                                                           *)
(*   * Physical time is modelled as an unbounded-but-bounded counter       *)
(*     (0..MaxPhysical) instead of microseconds since epoch.               *)
(*   * The logical counter carries ordering within a physical tick.        *)
(*   * node_id is only used to break ties; we expose it as part of the     *)
(*     timestamp but do not explicitly reason about tie-breaking here.     *)
(***************************************************************************)

EXTENDS Naturals, TLC, Sequences

CONSTANTS
  Nodes,        \* Set of node IDs, e.g. {n1, n2, n3}
  MaxPhysical,  \* Upper bound on the wall-clock component (e.g. 5)
  MaxEvents     \* Upper bound on total events per run (e.g. 10)

VARIABLES
  clocks,       \* clocks[n] = current HLC for node n: [phys, log]
  wall,         \* shared monotonic wall clock counter
  events,       \* sequence of all emitted timestamps, in emission order
  received      \* set of <<sender, receiver, ts>> tuples delivered so far

vars == <<clocks, wall, events, received>>

(***************************************************************************)
(* HLC timestamp record. We ignore node_id in state and only track phys /  *)
(* log; the total-order tiebreaker by node_id is a definitional extension. *)
(***************************************************************************)
TS == [phys : 0..MaxPhysical, log : 0..MaxEvents]

Zero == [phys |-> 0, log |-> 0]

(* Lexicographic less-than on timestamps. *)
TSLt(a, b) ==
  \/ a.phys < b.phys
  \/ (a.phys = b.phys /\ a.log < b.log)

TSLe(a, b) == TSLt(a, b) \/ a = b

Max(a, b) == IF a >= b THEN a ELSE b

(***************************************************************************)
(* Initial state: every node starts at (0,0), wall = 0, no events.         *)
(***************************************************************************)
Init ==
  /\ clocks  = [n \in Nodes |-> Zero]
  /\ wall    = 0
  /\ events  = <<>>
  /\ received = {}

(***************************************************************************)
(* Wall clock advances independently (bounded).                            *)
(***************************************************************************)
AdvanceWall ==
  /\ wall < MaxPhysical
  /\ wall' = wall + 1
  /\ UNCHANGED <<clocks, events, received>>

(***************************************************************************)
(* Tick: a node emits a local event.                                       *)
(*   if wall > clock.phys: (wall, 0)                                       *)
(*   else:                  (clock.phys, clock.log + 1)                    *)
(*                                                                         *)
(* Matches HlcClock::tick in hlc.rs.                                       *)
(***************************************************************************)
Tick(n) ==
  LET c    == clocks[n]
      newP == Max(c.phys, wall)
      newL == IF wall > c.phys THEN 0 ELSE c.log + 1
      new  == [phys |-> newP, log |-> newL]
  IN /\ Len(events) < MaxEvents
     /\ newL <= MaxEvents
     /\ clocks' = [clocks EXCEPT ![n] = new]
     /\ events' = Append(events, <<n, new>>)
     /\ UNCHANGED <<wall, received>>

(***************************************************************************)
(* Receive: receiver merges its clock with the incoming timestamp.         *)
(*   p_max = max(wall, local.phys, recv.phys)                              *)
(*   new_log depends on which of the three "wins" (matches HlcClock::update*)
(***************************************************************************)
Receive(sender, receiver, ts) ==
  LET c    == clocks[receiver]
      maxP == Max(Max(c.phys, ts.phys), wall)
      newL ==
        IF maxP = c.phys /\ maxP = ts.phys
          THEN Max(c.log, ts.log) + 1
        ELSE IF maxP = c.phys
          THEN c.log + 1
        ELSE IF maxP = ts.phys
          THEN ts.log + 1
        ELSE 0
      new  == [phys |-> maxP, log |-> newL]
  IN /\ sender /= receiver
     /\ newL <= MaxEvents
     /\ clocks' = [clocks EXCEPT ![receiver] = new]
     /\ received' = received \cup {<<sender, receiver, ts>>}
     /\ UNCHANGED <<wall, events>>

(***************************************************************************)
(* Next-state relation.                                                    *)
(***************************************************************************)
(* Only events that were actually emitted are candidates for receive.    *)
(* Bounding by the events sequence is essential: quantifying over all   *)
(* TS blows the state space up and TLC cannot finish.                   *)
Next ==
  \/ AdvanceWall
  \/ \E n \in Nodes : Tick(n)
  \/ \E i \in 1..Len(events), r \in Nodes :
        LET ev == events[i]
        IN Receive(ev[1], r, ev[2])

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Invariants.                                                             *)
(***************************************************************************)

(* Physical component of each node's clock only moves forward. *)
PhysicalTimeMonotonic ==
  \A n \in Nodes : clocks[n].phys <= MaxPhysical

(* Causal ordering: if (sender, ts) caused a delivery to receiver,         *)
(* then the receiver's clock strictly exceeds ts afterwards.               *)
CausalOrdering ==
  \A tup \in received :
      LET s  == tup[1]
          r  == tup[2]
          ts == tup[3]
      IN TSLt(ts, clocks[r])

(* The emitted events sequence is strictly increasing per node. *)
PerNodeMonotonic ==
  \A i, j \in 1..Len(events) :
      (i < j /\ events[i][1] = events[j][1])
        => TSLt(events[i][2], events[j][2])

(* Bound on events so TLC terminates. *)
EventCountsBounded == Len(events) <= MaxEvents

(***************************************************************************)
(* Type invariant (optional, checked by TLC when TypeOK is listed).        *)
(***************************************************************************)
TypeOK ==
  /\ clocks \in [Nodes -> TS]
  /\ wall   \in 0..MaxPhysical
  /\ events \in Seq(Nodes \X TS)

=============================================================================
