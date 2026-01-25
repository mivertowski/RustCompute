//! System memory utilities.

/// Gets available system memory in bytes.
///
/// Returns None if unable to determine.
#[must_use]
pub fn get_available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, we could use sysctl, but for simplicity return None
        // and let callers use conservative defaults
        None
    }

    #[cfg(target_os = "windows")]
    {
        // On Windows, we could use GlobalMemoryStatusEx, but for simplicity
        // return None and let callers use conservative defaults
        None
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Gets total system memory in bytes.
///
/// Returns None if unable to determine.
#[must_use]
pub fn get_total_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    {
        None
    }

    #[cfg(target_os = "windows")]
    {
        None
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

/// Gets system memory utilization (0.0-1.0).
///
/// Returns None if unable to determine.
#[must_use]
#[allow(dead_code)]
pub fn get_memory_utilization() -> Option<f64> {
    let total = get_total_memory()?;
    let available = get_available_memory()?;

    if total == 0 {
        return None;
    }

    let used = total.saturating_sub(available);
    Some(used as f64 / total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_available_memory() {
        // This may or may not return a value depending on platform
        let _memory = get_available_memory();
    }

    #[test]
    fn test_get_total_memory() {
        let _memory = get_total_memory();
    }

    #[test]
    fn test_get_memory_utilization() {
        if let Some(util) = get_memory_utilization() {
            assert!((0.0..=1.0).contains(&util));
        }
    }
}
