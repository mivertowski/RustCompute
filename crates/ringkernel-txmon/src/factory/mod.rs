//! Transaction factory for generating synthetic transactions.

mod accounts;
mod generator;
mod patterns;

pub use accounts::AccountGenerator;
pub use generator::{CustomerStats, FactoryState, GeneratorConfig, TransactionGenerator};
pub use patterns::TransactionPattern;
