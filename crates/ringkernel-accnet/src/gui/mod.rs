//! GUI components for accounting network visualization.
//!
//! Built with egui for cross-platform immediate mode rendering.
//!
//! ## Components
//!
//! - **NetworkCanvas**: Force-directed graph layout with flow animations
//! - **AnalyticsPanel**: Real-time metrics and risk indicators
//! - **AlertsFeed**: Scrolling feed of detected anomalies
//! - **EducationalOverlay**: Contextual explanations of accounting concepts
//! - **Charts**: Histograms, bar charts, sparklines for data visualization
//! - **Heatmaps**: Activity and correlation heatmaps
//! - **Rankings**: Top accounts by PageRank, centrality, risk

pub mod app;
pub mod canvas;
pub mod layout;
pub mod panels;
pub mod animation;
pub mod theme;
pub mod charts;
pub mod dashboard;
pub mod heatmaps;
pub mod rankings;

pub use app::AccNetApp;
pub use canvas::NetworkCanvas;
pub use layout::ForceDirectedLayout;
pub use panels::{AnalyticsPanel, ControlPanel, AlertsPanel};
pub use animation::{FlowParticle, ParticleSystem};
pub use theme::AccNetTheme;
pub use charts::{Histogram, BarChart, BalanceBarChart, DonutChart, Sparkline, MethodDistribution};
pub use heatmaps::{ActivityHeatmap, CorrelationHeatmap, RiskHeatmap, HeatmapGradient};
pub use rankings::{TopAccountsPanel, PatternStatsPanel, AmountDistribution, RankedAccount};
