//! Data fabric for synthetic process event generation.
//!
//! Provides configurable event streams for different industry sectors.

mod anomaly_injection;
mod event_gen;
mod pipeline;
mod process_models;
mod sector;

pub use anomaly_injection::*;
pub use event_gen::*;
pub use pipeline::*;
pub use process_models::*;
pub use sector::*;
