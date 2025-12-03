//! Loop transpilation for WGSL code generation.
//!
//! Handles Rust for/while/loop constructs and converts them to WGSL equivalents.

/// Represents recognized loop patterns from Rust DSL.
#[derive(Debug, Clone)]
pub enum LoopPattern {
    /// `for i in start..end` - exclusive range
    ForRange {
        var: String,
        start: String,
        end: String,
        inclusive: bool,
    },
    /// `while condition { ... }`
    While { condition: String },
    /// `loop { ... }` - infinite loop with break
    Loop,
}

impl LoopPattern {
    /// Generate the WGSL loop header.
    pub fn to_wgsl_header(&self) -> String {
        match self {
            LoopPattern::ForRange {
                var,
                start,
                end,
                inclusive,
            } => {
                let op = if *inclusive { "<=" } else { "<" };
                format!("for (var {var}: i32 = {start}; {var} {op} {end}; {var} = {var} + 1)")
            }
            LoopPattern::While { condition } => {
                format!("while ({condition})")
            }
            LoopPattern::Loop => "loop".to_string(),
        }
    }
}

/// Information about a range expression.
#[derive(Debug, Clone)]
pub struct RangeInfo {
    /// Start of range (or None for `..end`)
    pub start: Option<String>,
    /// End of range (or None for `start..`)
    pub end: Option<String>,
    /// Whether the range is inclusive (`..=`)
    pub inclusive: bool,
}

impl RangeInfo {
    /// Create a new range info.
    pub fn new(start: Option<String>, end: Option<String>, inclusive: bool) -> Self {
        Self {
            start,
            end,
            inclusive,
        }
    }

    /// Get the start expression, defaulting to "0" if not specified.
    pub fn start_or_default(&self) -> String {
        self.start.clone().unwrap_or_else(|| "0".to_string())
    }

    /// Get the end expression, or None if unbounded.
    pub fn end_expr(&self) -> Option<&str> {
        self.end.as_deref()
    }
}

/// Convert a Rust range to a WGSL for loop pattern.
pub fn range_to_for_loop(var: &str, range: &RangeInfo) -> LoopPattern {
    LoopPattern::ForRange {
        var: var.to_string(),
        start: range.start_or_default(),
        end: range
            .end
            .clone()
            .unwrap_or_else(|| "/* unbounded */".to_string()),
        inclusive: range.inclusive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_for_range_exclusive() {
        let pattern = LoopPattern::ForRange {
            var: "i".to_string(),
            start: "0".to_string(),
            end: "10".to_string(),
            inclusive: false,
        };
        assert_eq!(
            pattern.to_wgsl_header(),
            "for (var i: i32 = 0; i < 10; i = i + 1)"
        );
    }

    #[test]
    fn test_for_range_inclusive() {
        let pattern = LoopPattern::ForRange {
            var: "j".to_string(),
            start: "1".to_string(),
            end: "n".to_string(),
            inclusive: true,
        };
        assert_eq!(
            pattern.to_wgsl_header(),
            "for (var j: i32 = 1; j <= n; j = j + 1)"
        );
    }

    #[test]
    fn test_while_loop() {
        let pattern = LoopPattern::While {
            condition: "x > 0".to_string(),
        };
        assert_eq!(pattern.to_wgsl_header(), "while (x > 0)");
    }

    #[test]
    fn test_infinite_loop() {
        let pattern = LoopPattern::Loop;
        assert_eq!(pattern.to_wgsl_header(), "loop");
    }

    #[test]
    fn test_range_info() {
        let range = RangeInfo::new(Some("0".to_string()), Some("10".to_string()), false);
        let pattern = range_to_for_loop("i", &range);
        assert_eq!(
            pattern.to_wgsl_header(),
            "for (var i: i32 = 0; i < 10; i = i + 1)"
        );
    }
}
