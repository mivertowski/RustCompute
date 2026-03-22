# Phase 1: Crash Safety & Error Handling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate panic-on-error paths in production code. Replace 188 `Result<_, String>` with typed enums, reduce 1,152 `unwrap()` calls to <200, fix TLS placeholder.

**Architecture:** Each crate gets a `src/error.rs` with a thiserror-derived error enum (following the pattern established in `ringkernel-core/src/error.rs`). All public functions return typed `Result<T, CrateError>`. Codegen crates wrap `fmt::Error` for `writeln!` chains.

**Tech Stack:** thiserror 2.0 (workspace dependency), rustls-pemfile 2.1 (already in deps), Rust `?` operator for error propagation.

**Spec:** `docs/superpowers/specs/2026-03-22-production-readiness-design.md` (Phase 1, items 1.1-1.4)

---

## File Structure

### New files to create:
- `crates/ringkernel-accnet/src/error.rs` — AccNet error enum
- `crates/ringkernel-wavesim/src/error.rs` — WaveSim error enum
- `crates/ringkernel-wavesim3d/src/error.rs` — WaveSim3D error enum
- `crates/ringkernel-txmon/src/error.rs` — TxMon error enum
- `crates/ringkernel-procint/src/error.rs` — ProcInt error enum
- `crates/ringkernel-audio-fft/src/error.rs` — AudioFFT error enum

### Existing files to modify:
- `crates/ringkernel-core/src/tls.rs` — Implement PEM parsing in `parse_pem()`
- `crates/ringkernel-cli/src/commands/mod.rs` — Fix unwrap on line 30
- `crates/ringkernel-cuda-codegen/src/handler.rs` — Replace `writeln!().unwrap()` with `?`
- `crates/ringkernel-cuda-codegen/src/transpiler.rs` — Same
- `crates/ringkernel-cuda-codegen/src/stencil.rs` — Same
- `crates/ringkernel-cuda-codegen/src/ring_kernel.rs` — Same
- `crates/ringkernel-cuda-codegen/src/persistent_fdtd.rs` — Same
- `crates/ringkernel-cuda-codegen/src/reduction_intrinsics.rs` — Same
- `crates/ringkernel-cuda-codegen/src/shared.rs` — Same
- `crates/ringkernel-wgpu-codegen/src/transpiler.rs` — Same pattern
- `crates/ringkernel-wgpu-codegen/src/bindings.rs` — Same pattern
- All `lib.rs` files in crates getting new error types — add `mod error; pub use error::*;`

---

## Task 1: Fix CLI Input Validation (Warmup)

**Files:**
- Modify: `crates/ringkernel-cli/src/commands/mod.rs:25-35`

- [ ] **Step 1: Read the current code**

Read `crates/ringkernel-cli/src/commands/mod.rs` to understand full context.

- [ ] **Step 2: Fix the unwrap**

Replace:
```rust
if !name.chars().next().unwrap().is_alphabetic() && !name.starts_with('_') {
```

With:
```rust
if let Some(c) = name.chars().next() {
    if !c.is_alphabetic() && c != '_' {
        return Err("Project name must start with a letter or underscore".to_string());
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p ringkernel-cli`
Expected: All 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/ringkernel-cli/src/commands/mod.rs
git commit -m "fix: replace unwrap with pattern match in CLI project name validation"
```

---

## Task 2: TLS Certificate Loading Implementation

**Files:**
- Modify: `crates/ringkernel-core/src/tls.rs:600-616`
- Modify: `crates/ringkernel-core/Cargo.toml` (verify rustls-pemfile is available)

- [ ] **Step 1: Read current TLS implementation**

Read `crates/ringkernel-core/src/tls.rs` lines 580-650 and the `CertificateInfo` struct definition.

- [ ] **Step 2: Read Cargo.toml for deps**

Verify `rustls-pemfile = "2.1"` is present and behind the `tls` feature flag. Also check for `x509-parser` or equivalent for cert metadata extraction.

- [ ] **Step 3: Write failing test**

Add to the test module in `tls.rs`:
```rust
#[cfg(test)]
mod pem_tests {
    use super::*;

    // Self-signed test cert (generated for testing only)
    const TEST_CERT_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJALRiMLAh8KtKMAoGCCqGSM49BAMCMBExDzANBgNVBAMMBnRl
c3RjbjAeFw0yNjAxMDEwMDAwMDBaFw0yNzAxMDEwMDAwMDBaMBExDzANBgNV
BAMMBnRlc3RjbjBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABEVFz3kQk0R3
lMKg0ZsYYzuCqj2gZkwJf1m4p0Q3XN3MjmM8OBXlPmIKClHdLQF0N5bvJNw
rMd6sXTJjkVVV0ujIzAhMB8GA1UdEQQYMBaCBnRlc3RjboIMdGVzdGNuLmxv
Y2FsMAoGCCqGSM49BAMCA0gAMEUCIQDz9x7B1MnUTKj8A3BKF08VXL3FZlGi
P7Tq3c9BqgQb4AIgZK6M8d7V7g3c2wKDpMPkZ7I5w0BM3nRN+TLgLH5T8yc=
-----END CERTIFICATE-----";

    const TEST_KEY_PEM: &[u8] = b"-----BEGIN EC PRIVATE KEY-----
MHQCAQEEIOjN7HVJLlT3xJ3kCn3qI9jMGnhaD3a9QKNl3JShBJmcoAcGCCqG
SM49AwEHoUQDQgAERUXPeRCTRHeUwqDRmxhjO4KqPaBmTAl/WbinRDdc3cyO
Yzw4FeU+YgoKUd0tAXQ3lu8k3Csxx3qxdMmORVVXSw==
-----END EC PRIVATE KEY-----";

    #[test]
    fn test_parse_pem_returns_nonempty_cert() {
        let store = CertificateStore::new(CertStoreConfig::default());
        let entry = store.parse_pem(TEST_CERT_PEM, TEST_KEY_PEM, None, None).unwrap();
        assert!(!entry.cert_chain.is_empty(), "cert_chain should not be empty");
        assert!(!entry.private_key.is_empty(), "private_key should not be empty");
    }

    #[test]
    fn test_parse_pem_extracts_subject() {
        let store = CertificateStore::new(CertStoreConfig::default());
        let entry = store.parse_pem(TEST_CERT_PEM, TEST_KEY_PEM, None, None).unwrap();
        assert!(entry.info.subject_cn.contains("test"), "subject_cn should contain test domain");
    }

    #[test]
    fn test_parse_pem_invalid_data_returns_error() {
        let store = CertificateStore::new(CertStoreConfig::default());
        let result = store.parse_pem(b"not a cert", b"not a key", None, None);
        assert!(result.is_err(), "should fail on invalid PEM data");
    }
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cargo test -p ringkernel-core --features tls -- pem_tests`
Expected: FAIL — cert_chain is empty, subject_cn assertion fails.

- [ ] **Step 5: Implement PEM parsing**

Replace the placeholder `parse_pem` method body with:
```rust
pub fn parse_pem(
    &self,
    cert_pem: &[u8],
    key_pem: &[u8],
    cert_path: Option<PathBuf>,
    key_path: Option<PathBuf>,
) -> TlsResult<CertificateEntry> {
    use std::io::BufReader;

    // Parse certificate chain
    let mut cert_reader = BufReader::new(cert_pem);
    let cert_chain: Vec<Vec<u8>> = rustls_pemfile::certs(&mut cert_reader)
        .filter_map(|r| r.ok())
        .map(|c| c.to_vec())
        .collect();

    if cert_chain.is_empty() {
        return Err(TlsError::CertificateLoad("No certificates found in PEM data".into()));
    }

    // Parse private key (try PKCS8 first, then EC, then RSA)
    let mut key_reader = BufReader::new(key_pem);
    let private_key = rustls_pemfile::private_key(&mut key_reader)
        .map_err(|e| TlsError::CertificateLoad(format!("Failed to parse private key: {}", e)))?
        .ok_or_else(|| TlsError::CertificateLoad("No private key found in PEM data".into()))?
        .secret_der()
        .to_vec();

    // Extract certificate info from the first (leaf) certificate
    let info = Self::extract_cert_info(&cert_chain[0]);

    Ok(CertificateEntry {
        cert_chain,
        private_key,
        info,
        loaded_at: Instant::now(),
        cert_path,
        key_path,
    })
}
```

Also add helper method if not present:
```rust
fn extract_cert_info(der_bytes: &[u8]) -> CertificateInfo {
    // Best-effort metadata extraction from DER-encoded certificate
    // For full parsing, x509-parser could be used, but for now extract basics
    CertificateInfo {
        subject_cn: Self::extract_cn_from_der(der_bytes).unwrap_or_default(),
        issuer: String::new(),
        not_before: None,
        not_after: None,
        sans: Vec::new(),
        key_usage: Vec::new(),
        fingerprint: sha2_fingerprint(der_bytes),
    }
}
```

Note: Adapt field names and types to match the actual `CertificateInfo` struct definition. Read the struct first.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p ringkernel-core --features tls -- pem_tests`
Expected: All 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/ringkernel-core/src/tls.rs
git commit -m "feat: implement TLS PEM certificate parsing (replace placeholder)"
```

---

## Task 3: Typed Errors for Application Crates

This task creates error types for the 6 application/showcase crates that currently use `Result<_, String>`. Each follows the same pattern.

**Files:**
- Create: `crates/ringkernel-accnet/src/error.rs`
- Modify: `crates/ringkernel-accnet/src/lib.rs` — add `mod error; pub use error::*;`
- Modify: `crates/ringkernel-accnet/Cargo.toml` — ensure thiserror dependency

Repeat pattern for: wavesim, wavesim3d, txmon, procint, audio-fft.

### Task 3a: AccNet Error Type

- [ ] **Step 1: Read accnet lib.rs to understand current error patterns**

Run: `grep -n "Result<.*String>" crates/ringkernel-accnet/src/**/*.rs` to find all string error sites.

- [ ] **Step 2: Create error.rs**

Create `crates/ringkernel-accnet/src/error.rs`:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AccNetError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("kernel error: {0}")]
    Kernel(String),

    #[error("graph error: {0}")]
    Graph(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("actor error: {0}")]
    Actor(String),

    #[error("{0}")]
    Other(String),
}

impl From<String> for AccNetError {
    fn from(s: String) -> Self {
        AccNetError::Other(s)
    }
}

impl From<&str> for AccNetError {
    fn from(s: &str) -> Self {
        AccNetError::Other(s.to_string())
    }
}

pub type Result<T> = std::result::Result<T, AccNetError>;
```

Note: Read the actual source files first to determine exact error variants needed. The above is a template — adapt variant names to match actual error contexts found in grep results.

- [ ] **Step 3: Register module in lib.rs**

Add to top of `crates/ringkernel-accnet/src/lib.rs`:
```rust
pub mod error;
pub use error::{AccNetError, Result};
```

- [ ] **Step 4: Ensure thiserror in Cargo.toml**

Check if `thiserror` is already in accnet's `Cargo.toml`. If not, add:
```toml
thiserror = { workspace = true }
```

- [ ] **Step 5: Migrate callers**

Replace `Result<_, String>` return types with `Result<_>` (using the new type alias). Replace `.map_err(|e| e.to_string())` with `.map_err(AccNetError::Cuda)` (or appropriate variant). Use `?` operator where possible.

- [ ] **Step 6: Run tests**

Run: `cargo test -p ringkernel-accnet`
Expected: All tests pass (no behavior change, only error type change).

- [ ] **Step 7: Commit**

```bash
git add crates/ringkernel-accnet/
git commit -m "refactor(accnet): replace Result<_, String> with typed AccNetError enum"
```

### Task 3b: WaveSim Error Type

Same pattern as 3a. Create `crates/ringkernel-wavesim/src/error.rs` with `WaveSimError`.

- [ ] **Step 1: Grep for string errors** — `grep -rn "Result<.*String>" crates/ringkernel-wavesim/src/`
- [ ] **Step 2: Create error.rs** with variants matching actual error contexts
- [ ] **Step 3: Register module** in lib.rs
- [ ] **Step 4: Ensure thiserror dependency**
- [ ] **Step 5: Migrate callers**
- [ ] **Step 6: Run tests** — `cargo test -p ringkernel-wavesim`
- [ ] **Step 7: Commit** — `"refactor(wavesim): replace Result<_, String> with typed WaveSimError"`

### Task 3c: WaveSim3D Error Type

- [ ] **Step 1-7:** Same pattern. Create `WaveSim3dError` in `crates/ringkernel-wavesim3d/src/error.rs`.

### Task 3d: TxMon Error Type

- [ ] **Step 1-7:** Same pattern. Create `TxMonError` in `crates/ringkernel-txmon/src/error.rs`.

### Task 3e: ProcInt Error Type

- [ ] **Step 1-7:** Same pattern. Create `ProcIntError` in `crates/ringkernel-procint/src/error.rs`.

### Task 3f: AudioFFT Error Type

- [ ] **Step 1-7:** Same pattern. Create `AudioFftError` in `crates/ringkernel-audio-fft/src/error.rs`.

---

## Task 4: Unwrap Reduction — Codegen Crates (249 + ~30)

The codegen crates use `writeln!(code, ...).unwrap()` extensively. The fix is to change codegen functions to return `Result<String, fmt::Error>` and use `?`.

**Files:**
- Modify: `crates/ringkernel-cuda-codegen/src/handler.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/transpiler.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/stencil.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/ring_kernel.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/persistent_fdtd.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/reduction_intrinsics.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/shared.rs`
- Modify: `crates/ringkernel-wgpu-codegen/src/transpiler.rs`
- Modify: `crates/ringkernel-wgpu-codegen/src/bindings.rs`

### Task 4a: CUDA Codegen handler.rs

- [ ] **Step 1: Read handler.rs to understand scope**

Count `writeln!(...).unwrap()` instances. Understand return type of each function.

- [ ] **Step 2: Change function signatures**

For each function that contains `writeln!().unwrap()`:
- If it returns `String`, change to `Result<String, fmt::Error>`
- If it returns nothing, change to `Result<(), fmt::Error>`

- [ ] **Step 3: Replace `.unwrap()` with `?`**

Global find-replace within the file:
```
.unwrap()  ->  ?
```

For `writeln!` calls this is safe because `fmt::Error` is the only error type.

- [ ] **Step 4: Update callers**

Any function calling these updated functions needs to propagate the error or handle it.
- In production code: propagate with `?` up to the public API
- In test code: keep `.unwrap()` (tests should panic on errors)

- [ ] **Step 5: Run tests**

Run: `cargo test -p ringkernel-cuda-codegen`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/ringkernel-cuda-codegen/src/handler.rs
git commit -m "refactor(cuda-codegen): replace writeln unwraps with ? in handler"
```

### Task 4b-4g: Remaining CUDA Codegen Files

Repeat Task 4a pattern for each file:
- [ ] **4b: transpiler.rs** — Same pattern, one commit per file
- [ ] **4c: stencil.rs** — Same pattern
- [ ] **4d: ring_kernel.rs** — Same pattern
- [ ] **4e: persistent_fdtd.rs** — Same pattern
- [ ] **4f: reduction_intrinsics.rs** — Same pattern
- [ ] **4g: shared.rs** — Same pattern

Run full test suite after all files: `cargo test -p ringkernel-cuda-codegen`

### Task 4h: WGPU Codegen

- [ ] **Step 1: Apply same pattern to wgpu-codegen**

Files: `transpiler.rs`, `bindings.rs`
Same approach: `writeln!().unwrap()` -> `?`, update return types.

- [ ] **Step 2: Run tests**

Run: `cargo test -p ringkernel-wgpu-codegen`

- [ ] **Step 3: Commit**

```bash
git add crates/ringkernel-wgpu-codegen/
git commit -m "refactor(wgpu-codegen): replace writeln unwraps with ? in transpiler and bindings"
```

---

## Task 5: Unwrap Reduction — ringkernel-core (327 instances)

This is the largest single crate. Attack in submodule groups.

**Files:**
- Modify: Multiple files in `crates/ringkernel-core/src/`

- [ ] **Step 1: Identify all unwrap sites**

Run: `grep -rn '\.unwrap()' crates/ringkernel-core/src/ --include='*.rs' | grep -v '#\[cfg(test)\]' | grep -v 'mod tests' | head -50`

Group by file to determine priority order.

- [ ] **Step 2: Categorize unwraps**

For each unwrap, categorize:
- **Safe unwrap** (e.g., `"constant".parse::<u32>().unwrap()`) — leave as-is or use `expect()`
- **Lock unwrap** (e.g., `mutex.lock().unwrap()`) — replace with `.lock().map_err(|_| RingKernelError::Internal(...))?`
- **Option unwrap** (e.g., `map.get(key).unwrap()`) — replace with `.ok_or_else(|| RingKernelError::NotFound(...))?`
- **Result unwrap** (e.g., `channel.send(msg).unwrap()`) — replace with `?` or `.map_err()`

- [ ] **Step 3: Fix non-test unwraps by module**

Process each submodule file. For lock poisoning, use a consistent pattern:
```rust
let guard = self.inner.lock()
    .map_err(|_| RingKernelError::Internal("lock poisoned".into()))?;
```

- [ ] **Step 4: Run tests after each file**

Run: `cargo test -p ringkernel-core` after each file change.

- [ ] **Step 5: Commit per logical group**

```bash
git commit -m "refactor(core): reduce unwrap calls in queue/message modules"
git commit -m "refactor(core): reduce unwrap calls in runtime/lifecycle modules"
git commit -m "refactor(core): reduce unwrap calls in enterprise feature modules"
```

---

## Task 6: Unwrap Reduction — Remaining Crates

Apply the same categorize-and-fix approach to the remaining crates.

- [ ] **Step 1: ringkernel-wavesim (98 unwraps)**

Focus on simulation hot paths. Leave GUI/rendering unwraps as lower priority (panicking on render failure is often acceptable).

Run: `cargo test -p ringkernel-wavesim`
Commit: `"refactor(wavesim): reduce unwrap calls in simulation code"`

- [ ] **Step 2: ringkernel-ecosystem (38 unwraps)**

Focus on web framework integration paths (Axum handlers, gRPC services).

Run: `cargo test -p ringkernel-ecosystem --features "persistent,actix,tower,axum,grpc"`
Commit: `"refactor(ecosystem): reduce unwrap calls in framework integrations"`

- [ ] **Step 3: ringkernel-cpu (37 unwraps)**

Run: `cargo test -p ringkernel-cpu`
Commit: `"refactor(cpu): reduce unwrap calls in CPU backend"`

- [ ] **Step 4: ringkernel-cuda (~25 unwraps)**

These are often in GPU memory paths where errors are critical.

Run: `cargo test -p ringkernel-cuda`
Commit: `"refactor(cuda): reduce unwrap calls in CUDA backend"`

- [ ] **Step 5: ringkernel-graph (~20 unwraps)**

Run: `cargo test -p ringkernel-graph`
Commit: `"refactor(graph): reduce unwrap calls in graph algorithms"`

- [ ] **Step 6: ringkernel-metal (~15 unwraps)**

Run: `cargo test -p ringkernel-metal`
Commit: `"refactor(metal): reduce unwrap calls in Metal backend"`

- [ ] **Step 7: All remaining crates**

Sweep remaining crates for any non-test unwraps.

Run: `cargo test --workspace`
Commit: `"refactor: reduce remaining unwrap calls across workspace"`

---

## Task 7: Workspace-Wide Verification

- [ ] **Step 1: Count remaining unwraps**

Run: `grep -rn '\.unwrap()' crates/*/src/**/*.rs --include='*.rs' | grep -v test | grep -v '#\[cfg(test)\]' | wc -l`
Target: <200 remaining (from 1,152).

- [ ] **Step 2: Count remaining Result<_, String>**

Run: `grep -rn 'Result<.*String>' crates/*/src/**/*.rs --include='*.rs' | grep -v test | wc -l`
Target: 0 remaining (from 188).

- [ ] **Step 3: Full workspace test**

Run: `cargo test --workspace`
Expected: All 1,416+ tests pass. No regressions.

- [ ] **Step 4: Clippy check**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No new warnings.

- [ ] **Step 5: Final commit**

```bash
git commit -m "chore: Phase 1 complete — crash safety & error handling hardening"
```

---

## Success Criteria

1. **Zero** `Result<_, String>` in non-test production code
2. **<200** `unwrap()` calls in non-test production code (down from 1,152)
3. **TLS** `parse_pem()` returns real cert data (not empty vectors)
4. **All 1,416+ tests** pass with no regressions
5. **Clippy** passes with no new warnings
6. Each application crate has a typed error enum with thiserror
