//! Data fabric for synthetic accounting data generation.
//!
//! The fabric creates realistic synthetic financial data for simulation,
//! including company archetypes, chart of accounts templates, and
//! transaction generators with configurable patterns.

mod anomaly_injection;
mod chart_of_accounts;
mod company;
mod pipeline;
mod transaction_gen;

pub use anomaly_injection::*;
pub use chart_of_accounts::*;
pub use company::*;
pub use pipeline::*;
pub use transaction_gen::*;
