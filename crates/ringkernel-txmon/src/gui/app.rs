//! Main iced Application for transaction monitoring.

use super::theme::{colors, severity_color};
use crate::factory::{CustomerStats, FactoryState, GeneratorConfig, TransactionGenerator};
use crate::monitoring::MonitoringEngine;
use crate::types::{AlertType, MonitoringAlert, Transaction};

use iced::widget::{
    button, column, container, horizontal_rule, horizontal_space, row, scrollable, slider, text,
    vertical_space, Column,
};
use iced::{time, Alignment, Color, Element, Length, Subscription, Theme};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Maximum number of recent transactions to display.
const MAX_RECENT_TRANSACTIONS: usize = 50;
/// Maximum number of recent alerts to display.
const MAX_RECENT_ALERTS: usize = 30;

/// The main transaction monitoring application.
pub struct TxMonApp {
    // Factory state
    factory_state: FactoryState,
    generator: TransactionGenerator,

    // Monitoring state
    engine: MonitoringEngine,

    // Data buffers
    recent_transactions: VecDeque<Transaction>,
    recent_alerts: VecDeque<MonitoringAlert>,

    // Statistics
    total_transactions: u64,
    total_alerts: u64,
    flagged_transactions: u64, // Transactions that triggered at least one alert
    transactions_per_second: f32,
    alerts_per_second: f32,
    customer_stats: CustomerStats,

    // Performance tracking
    last_stats_update: Instant,
    transactions_since_update: u64,
    alerts_since_update: u64,

    // UI state
    transactions_per_second_slider: u32,
    suspicious_rate_slider: u8,
    selected_alert: Option<usize>,

    // Alert tracking per customer
    customer_alert_counts: HashMap<u64, u32>,
}

/// Messages for the application.
#[derive(Debug, Clone)]
pub enum Message {
    // Factory controls
    StartFactory,
    PauseFactory,
    StopFactory,
    SetTransactionRate(u32),
    SetSuspiciousRate(u8),

    // Processing
    Tick,

    // UI interactions
    SelectAlert(usize),
    ClearAlerts,
}

impl TxMonApp {
    /// Create a new application instance.
    pub fn new() -> Self {
        let config = GeneratorConfig::default();
        let generator = TransactionGenerator::new(config.clone());
        let customer_stats = generator.customer_stats();

        Self {
            factory_state: FactoryState::Stopped,
            generator,
            engine: MonitoringEngine::default(),
            recent_transactions: VecDeque::with_capacity(MAX_RECENT_TRANSACTIONS),
            recent_alerts: VecDeque::with_capacity(MAX_RECENT_ALERTS),
            total_transactions: 0,
            total_alerts: 0,
            flagged_transactions: 0,
            transactions_per_second: 0.0,
            alerts_per_second: 0.0,
            customer_stats,
            last_stats_update: Instant::now(),
            transactions_since_update: 0,
            alerts_since_update: 0,
            transactions_per_second_slider: config.transactions_per_second,
            suspicious_rate_slider: config.suspicious_rate,
            selected_alert: None,
            customer_alert_counts: HashMap::new(),
        }
    }

    /// Update the application state based on a message.
    pub fn update(&mut self, message: Message) {
        match message {
            Message::StartFactory => {
                self.factory_state = FactoryState::Running;
            }

            Message::PauseFactory => {
                self.factory_state = FactoryState::Paused;
            }

            Message::StopFactory => {
                self.factory_state = FactoryState::Stopped;
                self.transactions_per_second = 0.0;
                self.alerts_per_second = 0.0;
            }

            Message::SetTransactionRate(rate) => {
                self.transactions_per_second_slider = rate;
                let mut config = self.generator.config().clone();
                config.transactions_per_second = rate;
                self.generator.set_config(config);
            }

            Message::SetSuspiciousRate(rate) => {
                self.suspicious_rate_slider = rate;
                let mut config = self.generator.config().clone();
                config.suspicious_rate = rate;
                self.generator.set_config(config);
            }

            Message::Tick => {
                if self.factory_state == FactoryState::Running {
                    self.process_tick();
                }
                self.update_stats();
            }

            Message::SelectAlert(idx) => {
                self.selected_alert = Some(idx);
            }

            Message::ClearAlerts => {
                self.recent_alerts.clear();
                self.selected_alert = None;
            }
        }
    }

    /// Process a simulation tick.
    fn process_tick(&mut self) {
        // Calculate how many transactions to generate this tick
        // At 60 FPS, we need to generate TPS/60 transactions per tick
        let config = self.generator.config();
        let batch_size = (config.transactions_per_second as f32 / 60.0).ceil() as usize;
        let batch_size = batch_size.max(1).min(config.batch_size as usize);

        // Generate and process transactions
        let (transactions, profiles) = self.generator.generate_batch();
        let transactions: Vec<_> = transactions.into_iter().take(batch_size).collect();
        let profiles: Vec<_> = profiles.into_iter().take(batch_size).collect();

        let alerts = self.engine.process_batch(&transactions, &profiles);

        // Count unique transactions that triggered alerts and track per-customer
        let mut flagged_tx_ids: HashSet<u64> = HashSet::new();
        for alert in &alerts {
            flagged_tx_ids.insert(alert.transaction_id);
            // Track alerts per customer
            *self
                .customer_alert_counts
                .entry(alert.customer_id)
                .or_insert(0) += 1;
        }
        let flagged_count = flagged_tx_ids.len() as u64;

        // Update counters
        self.total_transactions += transactions.len() as u64;
        self.total_alerts += alerts.len() as u64;
        self.flagged_transactions += flagged_count;
        self.transactions_since_update += transactions.len() as u64;
        self.alerts_since_update += alerts.len() as u64;

        // Store recent transactions
        for tx in transactions {
            if self.recent_transactions.len() >= MAX_RECENT_TRANSACTIONS {
                self.recent_transactions.pop_front();
            }
            self.recent_transactions.push_back(tx);
        }

        // Store recent alerts (newest first)
        for alert in alerts {
            if self.recent_alerts.len() >= MAX_RECENT_ALERTS {
                self.recent_alerts.pop_back();
            }
            self.recent_alerts.push_front(alert);
        }
    }

    /// Update statistics (called every tick).
    fn update_stats(&mut self) {
        let elapsed = self.last_stats_update.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let secs = elapsed.as_secs_f32();
            self.transactions_per_second = self.transactions_since_update as f32 / secs;
            self.alerts_per_second = self.alerts_since_update as f32 / secs;

            self.transactions_since_update = 0;
            self.alerts_since_update = 0;
            self.last_stats_update = Instant::now();
        }
    }

    /// Build the view for the application.
    pub fn view(&self) -> Element<'_, Message> {
        let content = column![
            self.view_header(),
            horizontal_rule(1),
            row![self.view_left_panel(), self.view_right_panel(),].spacing(20),
        ]
        .padding(20)
        .spacing(15);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| container::Style {
                background: Some(colors::BACKGROUND.into()),
                ..Default::default()
            })
            .into()
    }

    /// View: Header with title and state indicator.
    fn view_header(&self) -> Element<'_, Message> {
        let state_color = match self.factory_state {
            FactoryState::Running => colors::STATE_RUNNING,
            FactoryState::Paused => colors::STATE_PAUSED,
            FactoryState::Stopped => colors::STATE_STOPPED,
        };

        row![
            text("RingKernel Transaction Monitor")
                .size(24)
                .color(colors::TEXT_PRIMARY),
            horizontal_space(),
            container(
                text(format!(" {} ", self.factory_state))
                    .size(14)
                    .color(colors::BACKGROUND)
            )
            .padding([4, 12])
            .style(move |_theme| container::Style {
                background: Some(state_color.into()),
                border: iced::Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }),
        ]
        .align_y(Alignment::Center)
        .into()
    }

    /// View: Left panel (controls, accounts, transactions).
    fn view_left_panel(&self) -> Element<'_, Message> {
        column![
            self.view_factory_controls(),
            vertical_space().height(15),
            self.view_account_overview(),
            vertical_space().height(15),
            self.view_transaction_feed(),
        ]
        .width(Length::FillPortion(3))
        .into()
    }

    /// View: Right panel (statistics, alerts).
    fn view_right_panel(&self) -> Element<'_, Message> {
        column![
            self.view_statistics(),
            vertical_space().height(10),
            self.view_alert_legend(),
            vertical_space().height(10),
            self.view_high_risk_accounts(),
            vertical_space().height(10),
            self.view_alerts_panel(),
        ]
        .width(Length::FillPortion(2))
        .into()
    }

    /// View: Factory controls panel.
    fn view_factory_controls(&self) -> Element<'_, Message> {
        let is_running = self.factory_state == FactoryState::Running;
        let is_stopped = self.factory_state == FactoryState::Stopped;

        let controls = row![
            button(text(if is_running { "Pause" } else { "Start" }).size(14))
                .padding([8, 16])
                .on_press(if is_running {
                    Message::PauseFactory
                } else {
                    Message::StartFactory
                }),
            button(text("Stop").size(14))
                .padding([8, 16])
                .on_press_maybe(if !is_stopped {
                    Some(Message::StopFactory)
                } else {
                    None
                }),
        ]
        .spacing(10);

        let rate_slider = row![
            text("Rate:").size(14).color(colors::TEXT_SECONDARY),
            slider(
                10..=5000,
                self.transactions_per_second_slider,
                Message::SetTransactionRate
            )
            .width(150),
            text(format!("{} tx/s", self.transactions_per_second_slider))
                .size(14)
                .color(colors::TEXT_PRIMARY),
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        let suspicious_slider = row![
            text("Suspicious:").size(14).color(colors::TEXT_SECONDARY),
            slider(
                0..=50,
                self.suspicious_rate_slider,
                Message::SetSuspiciousRate
            )
            .width(150),
            text(format!("{}%", self.suspicious_rate_slider))
                .size(14)
                .color(colors::TEXT_PRIMARY),
        ]
        .spacing(10)
        .align_y(Alignment::Center);

        self.panel(
            "Factory Controls",
            column![controls, rate_slider, suspicious_slider,].spacing(12),
        )
    }

    /// View: Account overview panel.
    fn view_account_overview(&self) -> Element<'_, Message> {
        let stats = &self.customer_stats;

        let content = column![
            row![
                self.stat_item("Active", format!("{}", stats.total)),
                self.stat_item("Low Risk", format!("{}", stats.low_risk)),
                self.stat_item("Medium", format!("{}", stats.medium_risk)),
            ]
            .spacing(20),
            row![
                self.stat_item_colored(
                    "High Risk",
                    format!("{}", stats.high_risk),
                    colors::ALERT_HIGH
                ),
                self.stat_item_colored("PEP", format!("{}", stats.pep), colors::ALERT_MEDIUM),
                self.stat_item_colored(
                    "EDD Req.",
                    format!("{}", stats.edd_required),
                    colors::ALERT_LOW
                ),
            ]
            .spacing(20),
        ]
        .spacing(10);

        self.panel("Account Overview", content)
    }

    /// View: Transaction feed panel.
    fn view_transaction_feed(&self) -> Element<'_, Message> {
        // Collect transactions to owned copies to avoid lifetime issues
        let txs: Vec<_> = self
            .recent_transactions
            .iter()
            .rev()
            .take(15)
            .copied()
            .collect();

        let transactions: Vec<Element<Message>> = txs
            .into_iter()
            .map(|tx| Self::view_transaction_row_static(tx))
            .collect();

        let feed = if transactions.is_empty() {
            column![text("No transactions yet...")
                .size(14)
                .color(colors::TEXT_DISABLED)]
        } else {
            Column::with_children(transactions).spacing(4)
        };

        self.panel("Live Transaction Feed", scrollable(feed).height(200))
    }

    /// View: Single transaction row (static version for use in closures).
    fn view_transaction_row_static(tx: Transaction) -> Element<'static, Message> {
        let time_str = format_timestamp(tx.timestamp);
        let amount_color = if tx.amount_cents >= 1_000_000 {
            colors::ALERT_HIGH
        } else {
            colors::TEXT_PRIMARY
        };

        row![
            text(time_str)
                .size(12)
                .color(colors::TEXT_SECONDARY)
                .width(70),
            text(format!("#{}", tx.transaction_id % 10000))
                .size(12)
                .color(colors::TEXT_SECONDARY)
                .width(50),
            text(tx.format_amount())
                .size(12)
                .color(amount_color)
                .width(80),
            text(format!("-> {}", tx.country_name()))
                .size(12)
                .color(colors::TEXT_SECONDARY),
            horizontal_space(),
            text(if tx.is_high_value() { "!" } else { "" })
                .size(12)
                .color(colors::ALERT_HIGH),
        ]
        .spacing(8)
        .align_y(Alignment::Center)
        .into()
    }

    /// View: Statistics panel.
    fn view_statistics(&self) -> Element<'_, Message> {
        // Flagged rate = percentage of transactions that triggered at least one alert
        let flagged_rate = if self.total_transactions > 0 {
            format!(
                "{:.2}%",
                (self.flagged_transactions as f64 / self.total_transactions as f64) * 100.0
            )
        } else {
            "0%".to_string()
        };

        let content = column![
            self.stat_row("TPS", format!("{:.0}", self.transactions_per_second)),
            self.stat_row("Alerts/s", format!("{:.1}", self.alerts_per_second)),
            horizontal_rule(1),
            self.stat_row("Total Processed", format_number(self.total_transactions)),
            self.stat_row("Total Alerts", format_number(self.total_alerts)),
            self.stat_row("Flagged Tx", format_number(self.flagged_transactions)),
            horizontal_rule(1),
            self.stat_row("Flagged Rate", flagged_rate),
        ]
        .spacing(8);

        self.panel("Statistics", content)
    }

    /// View: Alert types legend.
    fn view_alert_legend(&self) -> Element<'_, Message> {
        let alert_types = [
            (
                AlertType::VelocityBreach,
                "Too many transactions in time window",
            ),
            (
                AlertType::AmountThreshold,
                "Transaction exceeds $ threshold",
            ),
            (
                AlertType::StructuredTransaction,
                "Smurfing: amounts just under threshold",
            ),
            (AlertType::GeographicAnomaly, "Unusual destination country"),
        ];

        let legend_items: Vec<Element<'_, Message>> = alert_types
            .iter()
            .map(|(alert_type, description)| {
                row![
                    text(alert_type.code()).size(11).color(colors::ACCENT_BLUE),
                    text(format!(" - {}", description))
                        .size(11)
                        .color(colors::TEXT_SECONDARY),
                ]
                .into()
            })
            .collect();

        self.panel(
            "Alert Types",
            Column::with_children(legend_items).spacing(4),
        )
    }

    /// View: Top 5 high-risk accounts.
    fn view_high_risk_accounts(&self) -> Element<'_, Message> {
        // Get top 5 customers by alert count
        let mut sorted_customers: Vec<_> = self.customer_alert_counts.iter().collect();
        sorted_customers.sort_by(|a, b| b.1.cmp(a.1));

        let top_accounts: Vec<Element<'_, Message>> = sorted_customers
            .iter()
            .take(5)
            .enumerate()
            .map(|(rank, (customer_id, alert_count))| {
                let rank_color = match rank {
                    0 => colors::ALERT_CRITICAL,
                    1 => colors::ALERT_HIGH,
                    2 => colors::ALERT_MEDIUM,
                    _ => colors::TEXT_SECONDARY,
                };
                row![
                    text(format!("{}.", rank + 1))
                        .size(12)
                        .color(rank_color)
                        .width(20),
                    text(format!("Customer #{}", customer_id))
                        .size(12)
                        .color(colors::TEXT_PRIMARY),
                    horizontal_space(),
                    text(format!("{} alerts", alert_count))
                        .size(12)
                        .color(rank_color),
                ]
                .spacing(8)
                .into()
            })
            .collect();

        let content = if top_accounts.is_empty() {
            column![text("No flagged accounts yet")
                .size(12)
                .color(colors::TEXT_DISABLED)]
        } else {
            Column::with_children(top_accounts).spacing(6)
        };

        self.panel("Top 5 High-Risk Accounts", content)
    }

    /// View: Alerts panel.
    fn view_alerts_panel(&self) -> Element<'_, Message> {
        let header = row![
            text("Compliance Alerts")
                .size(16)
                .color(colors::TEXT_PRIMARY),
            horizontal_space(),
            button(text("Clear").size(12))
                .padding([4, 8])
                .on_press(Message::ClearAlerts),
        ]
        .align_y(Alignment::Center);

        let alerts: Vec<Element<Message>> = self
            .recent_alerts
            .iter()
            .enumerate()
            .take(10)
            .map(|(idx, alert)| self.view_alert_row(idx, alert))
            .collect();

        let alerts_list = if alerts.is_empty() {
            column![text("No alerts").size(14).color(colors::TEXT_DISABLED)]
        } else {
            Column::with_children(alerts).spacing(8)
        };

        container(
            column![
                header,
                vertical_space().height(10),
                scrollable(alerts_list).height(200),
            ]
            .spacing(5),
        )
        .padding(15)
        .width(Length::Fill)
        .style(|_theme| container::Style {
            background: Some(colors::SURFACE.into()),
            border: iced::Border {
                color: colors::BORDER,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        })
        .into()
    }

    /// View: Single alert row.
    fn view_alert_row(&self, idx: usize, alert: &MonitoringAlert) -> Element<'_, Message> {
        let severity = alert.severity();
        let alert_type = alert.alert_type();
        let color = severity_color(severity);

        let indicator = text(severity.indicator().to_string()).size(16).color(color);

        let is_selected = self.selected_alert == Some(idx);
        let bg_color = if is_selected {
            colors::SURFACE_BRIGHT
        } else {
            colors::SURFACE
        };

        let content = column![
            row![
                indicator,
                text(severity.name()).size(12).color(color),
                horizontal_space(),
                text(format!("#{}", alert.alert_id % 10000))
                    .size(10)
                    .color(colors::TEXT_DISABLED),
            ]
            .spacing(8)
            .align_y(Alignment::Center),
            text(alert_type.name()).size(14).color(colors::TEXT_PRIMARY),
            row![
                text(format!("Cust: {}", alert.customer_id))
                    .size(11)
                    .color(colors::TEXT_SECONDARY),
                text(alert.format_amount())
                    .size(11)
                    .color(colors::TEXT_SECONDARY),
            ]
            .spacing(15),
        ]
        .spacing(4);

        button(content)
            .padding(10)
            .width(Length::Fill)
            .on_press(Message::SelectAlert(idx))
            .style(move |_theme, _status| button::Style {
                background: Some(bg_color.into()),
                border: iced::Border {
                    color: color.scale_alpha(0.3),
                    width: 1.0,
                    radius: 6.0.into(),
                },
                text_color: colors::TEXT_PRIMARY,
                ..Default::default()
            })
            .into()
    }

    /// Helper: Create a panel with title.
    fn panel(
        &self,
        title: &'static str,
        content: impl Into<Element<'static, Message>>,
    ) -> Element<'static, Message> {
        container(
            column![
                text(title).size(16).color(colors::TEXT_PRIMARY),
                vertical_space().height(10),
                content.into(),
            ]
            .spacing(5),
        )
        .padding(15)
        .width(Length::Fill)
        .style(|_theme| container::Style {
            background: Some(colors::SURFACE.into()),
            border: iced::Border {
                color: colors::BORDER,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        })
        .into()
    }

    /// Helper: Create a stat item.
    fn stat_item(&self, label: &'static str, value: String) -> Element<'static, Message> {
        column![
            text(label).size(11).color(colors::TEXT_SECONDARY),
            text(value).size(16).color(colors::TEXT_PRIMARY),
        ]
        .spacing(2)
        .into()
    }

    /// Helper: Create a colored stat item.
    fn stat_item_colored(
        &self,
        label: &'static str,
        value: String,
        color: Color,
    ) -> Element<'static, Message> {
        column![
            text(label).size(11).color(colors::TEXT_SECONDARY),
            text(value).size(16).color(color),
        ]
        .spacing(2)
        .into()
    }

    /// Helper: Create a stat row.
    fn stat_row(&self, label: &'static str, value: String) -> Element<'static, Message> {
        row![
            text(label).size(14).color(colors::TEXT_SECONDARY),
            horizontal_space(),
            text(value).size(14).color(colors::TEXT_PRIMARY),
        ]
        .into()
    }

    /// Get the theme for the application.
    pub fn theme(&self) -> Theme {
        Theme::Dark
    }

    /// Get subscriptions for the application.
    pub fn subscription(&self) -> Subscription<Message> {
        // 60 FPS tick for animation and processing
        time::every(Duration::from_millis(16)).map(|_| Message::Tick)
    }
}

impl Default for TxMonApp {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a timestamp as HH:MM:SS.
fn format_timestamp(timestamp_ms: u64) -> String {
    let secs = (timestamp_ms / 1000) % 86400;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

/// Format a large number with K/M suffixes.
fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
