//! Dark banking theme colors.

use iced::Color;

/// Color palette for the application.
pub mod colors {
    use super::*;

    /// Main background color.
    pub const BACKGROUND: Color = Color::from_rgb(0.08, 0.09, 0.11);

    /// Surface color for cards/panels.
    pub const SURFACE: Color = Color::from_rgb(0.12, 0.13, 0.16);

    /// Brighter surface for hover/active states.
    pub const SURFACE_BRIGHT: Color = Color::from_rgb(0.18, 0.19, 0.22);

    /// Border color.
    pub const BORDER: Color = Color::from_rgb(0.25, 0.26, 0.30);

    /// Primary text color.
    pub const TEXT_PRIMARY: Color = Color::from_rgb(0.92, 0.92, 0.94);

    /// Secondary/muted text color.
    pub const TEXT_SECONDARY: Color = Color::from_rgb(0.6, 0.62, 0.66);

    /// Disabled text color.
    pub const TEXT_DISABLED: Color = Color::from_rgb(0.4, 0.42, 0.46);

    /// Primary accent color (blue).
    pub const ACCENT_BLUE: Color = Color::from_rgb(0.2, 0.5, 0.9);

    /// Success/positive color (green).
    pub const ACCENT_GREEN: Color = Color::from_rgb(0.2, 0.8, 0.4);

    /// Warning color (orange).
    pub const ACCENT_ORANGE: Color = Color::from_rgb(0.9, 0.5, 0.1);

    /// Critical alert color (red).
    pub const ALERT_CRITICAL: Color = Color::from_rgb(0.9, 0.2, 0.2);

    /// High alert color (orange).
    pub const ALERT_HIGH: Color = Color::from_rgb(0.9, 0.5, 0.1);

    /// Medium alert color (yellow).
    pub const ALERT_MEDIUM: Color = Color::from_rgb(0.9, 0.8, 0.2);

    /// Low alert color (green).
    pub const ALERT_LOW: Color = Color::from_rgb(0.4, 0.7, 0.4);

    /// Money positive (incoming).
    pub const MONEY_POSITIVE: Color = Color::from_rgb(0.2, 0.8, 0.4);

    /// Money negative (outgoing).
    pub const MONEY_NEGATIVE: Color = Color::from_rgb(0.9, 0.3, 0.3);

    /// Running state color.
    pub const STATE_RUNNING: Color = Color::from_rgb(0.2, 0.8, 0.4);

    /// Paused state color.
    pub const STATE_PAUSED: Color = Color::from_rgb(0.9, 0.8, 0.2);

    /// Stopped state color.
    pub const STATE_STOPPED: Color = Color::from_rgb(0.6, 0.62, 0.66);
}

/// Get color for alert severity.
pub fn severity_color(severity: crate::types::AlertSeverity) -> Color {
    match severity {
        crate::types::AlertSeverity::Low => colors::ALERT_LOW,
        crate::types::AlertSeverity::Medium => colors::ALERT_MEDIUM,
        crate::types::AlertSeverity::High => colors::ALERT_HIGH,
        crate::types::AlertSeverity::Critical => colors::ALERT_CRITICAL,
    }
}
