//! Data fabric for synthetic accounting data generation.
//!
//! The fabric creates realistic synthetic financial data for simulation,
//! including company archetypes, chart of accounts templates, and
//! transaction generators with configurable patterns.

mod company;
mod chart_of_accounts;
mod transaction_gen;
mod anomaly_injection;
mod pipeline;

pub use company::*;
pub use chart_of_accounts::*;
pub use transaction_gen::*;
pub use anomaly_injection::*;
pub use pipeline::*;
