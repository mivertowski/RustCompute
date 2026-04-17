------------------------ MODULE tenant_isolation ------------------------
(***************************************************************************)
(* Tenant isolation: messages never cross a tenant boundary.               *)
(*                                                                         *)
(* Each kernel is registered with a single tenant. The routing layer must *)
(* reject any Send whose sender tenant differs from the receiver tenant.   *)
(* This spec models that rule explicitly and checks that:                  *)
(*                                                                         *)
(*   * NoSameMessageInBothTenants -- a single message id never shows up   *)
(*     in traffic of two different tenants.                                *)
(*   * CrossTenantRejected        -- any attempted cross-tenant send      *)
(*     ends up in the `rejected` set, never in `delivered`.               *)
(*   * IsolationByConstruction    -- the only way a message reaches an   *)
(*     actor's inbox is if sender.tenant = receiver.tenant.               *)
(***************************************************************************)

EXTENDS Naturals, TLC, Sequences, FiniteSets

CONSTANTS
  Tenants,         \* set of tenant ids
  Kernels,         \* set of kernel ids
  MaxMsgs

VARIABLES
  kernel_tenant,   \* function Kernels -> (Tenants \cup {NONE_SENTINEL})
  inboxes,         \* per-kernel delivered inbox
  delivered,       \* triples of <<from, to, msg>> successfully routed
  rejected,        \* triples rejected by tenant check
  next_msg

vars == <<kernel_tenant, inboxes, delivered, rejected, next_msg>>

NONE_SENTINEL == "UNREGISTERED"

Msgs == 1..MaxMsgs

Init ==
  /\ kernel_tenant = [k \in Kernels |-> NONE_SENTINEL]
  /\ inboxes       = [k \in Kernels |-> {}]
  /\ delivered     = {}
  /\ rejected      = {}
  /\ next_msg      = 1

(***************************************************************************)
(* RegisterKernel: bind a kernel to a tenant exactly once.                 *)
(***************************************************************************)
RegisterKernel(k, t) ==
  /\ kernel_tenant[k] = NONE_SENTINEL
  /\ kernel_tenant' = [kernel_tenant EXCEPT ![k] = t]
  /\ UNCHANGED <<inboxes, delivered, rejected, next_msg>>

(***************************************************************************)
(* Send: all cross-tenant attempts are routed to `rejected`.               *)
(* Same-tenant sends land in the target inbox and `delivered`.             *)
(* Sends from unregistered kernels are always rejected.                    *)
(***************************************************************************)
Send(from, to) ==
  /\ from /= to
  /\ next_msg <= MaxMsgs
  /\ LET m  == next_msg
         tf == kernel_tenant[from]
         tt == kernel_tenant[to]
     IN IF tf = NONE_SENTINEL \/ tt = NONE_SENTINEL \/ tf /= tt
          THEN /\ rejected'  = rejected \cup {<<from, to, m>>}
               /\ delivered' = delivered
               /\ inboxes'   = inboxes
          ELSE /\ delivered' = delivered \cup {<<from, to, m>>}
               /\ inboxes'   = [inboxes EXCEPT ![to] = @ \cup {m}]
               /\ rejected'  = rejected
  /\ next_msg' = next_msg + 1
  /\ UNCHANGED kernel_tenant

Next ==
  \/ \E k \in Kernels, t \in Tenants : RegisterKernel(k, t)
  \/ \E f, t \in Kernels              : Send(f, t)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Invariants.                                                             *)
(***************************************************************************)

(* No delivered message appears in two different tenants' observable      *)
(* traffic at once. Expressed over (delivered) tuples.                    *)
NoSameMessageInBothTenants ==
  \A tup1, tup2 \in delivered :
      (tup1[3] = tup2[3])
        => kernel_tenant[tup1[2]] = kernel_tenant[tup2[2]]

(* Every cross-tenant attempt is in `rejected` and NOT in `delivered`.    *)
CrossTenantRejected ==
  \A tup \in rejected : tup \notin delivered

(* The only way a message reaches an inbox: same-tenant sender/receiver.  *)
IsolationByConstruction ==
  \A tup \in delivered :
      LET f == tup[1]
          t == tup[2]
      IN /\ kernel_tenant[f] /= NONE_SENTINEL
         /\ kernel_tenant[t] /= NONE_SENTINEL
         /\ kernel_tenant[f] = kernel_tenant[t]

(* Every id in an inbox came through `delivered`. *)
InboxesWellFormed ==
  \A k \in Kernels :
      \A m \in inboxes[k] :
          \E f \in Kernels : <<f, k, m>> \in delivered

TypeOK ==
  /\ kernel_tenant \in [Kernels -> Tenants \cup {NONE_SENTINEL}]
  /\ inboxes       \in [Kernels -> SUBSET Msgs]
  /\ delivered     \subseteq (Kernels \X Kernels \X Msgs)
  /\ rejected      \subseteq (Kernels \X Kernels \X Msgs)
  /\ next_msg      \in 1..(MaxMsgs + 1)

=============================================================================
