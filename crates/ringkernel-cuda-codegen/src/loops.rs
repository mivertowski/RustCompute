//! Loop transpilation helpers for CUDA code generation.
//!
//! This module provides utilities for transpiling Rust loop constructs
//! to their CUDA C equivalents.
//!
//! # Supported Loop Types
//!
//! - `for i in start..end` → `for (int i = start; i < end; i++)`
//! - `for i in start..=end` → `for (int i = start; i <= end; i++)`
//! - `while condition` → `while (condition)`
//! - `loop` → `while (true)` or `for (;;)`
//!
//! # Control Flow
//!
//! - `break` → `break;`
//! - `continue` → `continue;`
//! - `break 'label` → Not yet supported (would require goto)

use syn::{Expr, ExprRange, RangeLimits};

/// Information about a parsed range expression.
#[derive(Debug, Clone)]
pub struct RangeInfo {
    /// Start value of the range (None for unbounded start).
    pub start: Option<String>,
    /// End value of the range (None for unbounded end).
    pub end: Option<String>,
    /// Whether the range is inclusive (..=) or exclusive (..).
    pub inclusive: bool,
}

impl RangeInfo {
    /// Parse a range expression into RangeInfo.
    ///
    /// # Arguments
    ///
    /// * `range` - The range expression to parse
    /// * `transpile_expr` - Function to transpile sub-expressions to CUDA strings
    ///
    /// # Returns
    ///
    /// A RangeInfo struct containing the parsed range bounds.
    pub fn from_range<F>(range: &ExprRange, transpile_expr: F) -> Self
    where
        F: Fn(&Expr) -> Result<String, crate::TranspileError>,
    {
        let start = range
            .start
            .as_ref()
            .and_then(|e| transpile_expr(e).ok());

        let end = range
            .end
            .as_ref()
            .and_then(|e| transpile_expr(e).ok());

        let inclusive = matches!(range.limits, RangeLimits::Closed(_));

        RangeInfo {
            start,
            end,
            inclusive,
        }
    }

    /// Generate the CUDA comparison operator for the loop condition.
    pub fn comparison_op(&self) -> &'static str {
        if self.inclusive {
            "<="
        } else {
            "<"
        }
    }

    /// Generate a complete CUDA for loop header.
    ///
    /// # Arguments
    ///
    /// * `var_name` - The loop variable name
    /// * `var_type` - The CUDA type for the loop variable (e.g., "int")
    ///
    /// # Returns
    ///
    /// A string like `for (int i = 0; i < n; i++)`
    pub fn to_cuda_for_header(&self, var_name: &str, var_type: &str) -> String {
        let start = self.start.as_deref().unwrap_or("0");
        let end = self.end.as_deref().unwrap_or("/* end */");
        let op = self.comparison_op();

        format!(
            "for ({var_type} {var_name} = {start}; {var_name} {op} {end}; {var_name}++)"
        )
    }
}

/// Represents different loop patterns that can be transpiled.
#[derive(Debug, Clone)]
pub enum LoopPattern {
    /// A for loop over a range: `for i in start..end`
    ForRange {
        var_name: String,
        range: RangeInfo,
    },
    /// A for loop over an iterator (not fully supported yet)
    ForIterator {
        var_name: String,
        iterator: String,
    },
    /// A while loop: `while condition { ... }`
    While {
        condition: String,
    },
    /// An infinite loop: `loop { ... }`
    Infinite,
}

impl LoopPattern {
    /// Generate the CUDA loop header for this pattern.
    ///
    /// # Arguments
    ///
    /// * `var_type` - The type to use for loop variables (e.g., "int")
    ///
    /// # Returns
    ///
    /// The CUDA loop header string.
    pub fn to_cuda_header(&self, var_type: &str) -> String {
        match self {
            LoopPattern::ForRange { var_name, range } => {
                range.to_cuda_for_header(var_name, var_type)
            }
            LoopPattern::ForIterator { var_name, iterator } => {
                // Basic iterator support - treat as range-like
                format!("for ({var_type} {var_name} : {iterator})")
            }
            LoopPattern::While { condition } => {
                format!("while ({condition})")
            }
            LoopPattern::Infinite => {
                // Using while(true) for clarity; could also use for(;;)
                "while (true)".to_string()
            }
        }
    }
}

/// Check if an expression is a simple range (start..end or start..=end).
pub fn is_range_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Range(_))
}

/// Extract the loop variable name from a for loop pattern.
pub fn extract_loop_var(pat: &syn::Pat) -> Option<String> {
    match pat {
        syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
        _ => None,
    }
}

/// Determine the appropriate CUDA type for a loop variable.
///
/// This uses heuristics based on the range bounds to pick int vs size_t.
pub fn infer_loop_var_type(range: &RangeInfo) -> &'static str {
    // Check if the range bounds suggest a specific type
    if let Some(ref end) = range.end {
        // If the end contains "size" or looks like a size type, use size_t
        if end.contains("size") || end.contains("len") {
            return "size_t";
        }
    }

    // Default to int for most cases
    "int"
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_range_info_exclusive() {
        let range: ExprRange = parse_quote!(0..10);
        let info = RangeInfo::from_range(&range, |e| {
            Ok(quote::ToTokens::to_token_stream(e).to_string())
        });

        assert!(!info.inclusive);
        assert_eq!(info.comparison_op(), "<");
    }

    #[test]
    fn test_range_info_inclusive() {
        let range: ExprRange = parse_quote!(0..=10);
        let info = RangeInfo::from_range(&range, |e| {
            Ok(quote::ToTokens::to_token_stream(e).to_string())
        });

        assert!(info.inclusive);
        assert_eq!(info.comparison_op(), "<=");
    }

    #[test]
    fn test_for_header_generation() {
        let range = RangeInfo {
            start: Some("0".to_string()),
            end: Some("n".to_string()),
            inclusive: false,
        };

        let header = range.to_cuda_for_header("i", "int");
        assert_eq!(header, "for (int i = 0; i < n; i++)");
    }

    #[test]
    fn test_for_header_inclusive() {
        let range = RangeInfo {
            start: Some("1".to_string()),
            end: Some("10".to_string()),
            inclusive: true,
        };

        let header = range.to_cuda_for_header("j", "int");
        assert_eq!(header, "for (int j = 1; j <= 10; j++)");
    }

    #[test]
    fn test_loop_pattern_while() {
        let pattern = LoopPattern::While {
            condition: "i < 10".to_string(),
        };

        assert_eq!(pattern.to_cuda_header("int"), "while (i < 10)");
    }

    #[test]
    fn test_loop_pattern_infinite() {
        let pattern = LoopPattern::Infinite;
        assert_eq!(pattern.to_cuda_header("int"), "while (true)");
    }

    #[test]
    fn test_extract_loop_var() {
        let pat: syn::Pat = parse_quote!(i);
        assert_eq!(extract_loop_var(&pat), Some("i".to_string()));

        let pat_complex: syn::Pat = parse_quote!((a, b));
        assert_eq!(extract_loop_var(&pat_complex), None);
    }

    #[test]
    fn test_infer_loop_var_type() {
        let range_int = RangeInfo {
            start: Some("0".to_string()),
            end: Some("10".to_string()),
            inclusive: false,
        };
        assert_eq!(infer_loop_var_type(&range_int), "int");

        let range_size = RangeInfo {
            start: Some("0".to_string()),
            end: Some("data.len()".to_string()),
            inclusive: false,
        };
        assert_eq!(infer_loop_var_type(&range_size), "size_t");
    }
}
