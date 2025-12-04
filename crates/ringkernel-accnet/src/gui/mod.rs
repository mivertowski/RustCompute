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

pub mod animation;
pub mod app;
pub mod canvas;
pub mod charts;
pub mod dashboard;
pub mod heatmaps;
pub mod layout;
pub mod panels;
pub mod rankings;
pub mod theme;

pub use animation::{FlowParticle, ParticleSystem};
pub use app::AccNetApp;
pub use canvas::NetworkCanvas;
pub use charts::{BalanceBarChart, BarChart, DonutChart, Histogram, MethodDistribution, Sparkline};
pub use heatmaps::{ActivityHeatmap, CorrelationHeatmap, HeatmapGradient, RiskHeatmap};
pub use layout::ForceDirectedLayout;
pub use panels::{AlertsPanel, AnalyticsPanel, ControlPanel};
pub use rankings::{AmountDistribution, PatternStatsPanel, RankedAccount, TopAccountsPanel};
pub use theme::AccNetTheme;
