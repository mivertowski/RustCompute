//! Temporal analysis structures for time-series based detection.
//!
//! These models support seasonality detection, trend analysis, and
//! behavioral anomaly identification.

use super::HybridTimestamp;
use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

/// Time granularity for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum TimeGranularity {
    /// Daily granularity (365 periods/year).
    Daily = 0,
    /// Weekly granularity (52 periods/year).
    Weekly = 1,
    /// Bi-weekly granularity (26 periods/year).
    BiWeekly = 2,
    /// Monthly granularity (12 periods/year).
    Monthly = 3,
    /// Quarterly granularity (4 periods/year).
    Quarterly = 4,
    /// Annual granularity (1 period/year).
    Annual = 5,
}

impl TimeGranularity {
    /// Periods per year for this granularity.
    pub fn periods_per_year(&self) -> u32 {
        match self {
            TimeGranularity::Daily => 365,
            TimeGranularity::Weekly => 52,
            TimeGranularity::BiWeekly => 26,
            TimeGranularity::Monthly => 12,
            TimeGranularity::Quarterly => 4,
            TimeGranularity::Annual => 1,
        }
    }

    /// Expected autocorrelation lag for seasonal detection.
    pub fn seasonal_lag(&self) -> u32 {
        match self {
            TimeGranularity::Daily => 7,     // Weekly pattern
            TimeGranularity::Weekly => 52,   // Annual pattern
            TimeGranularity::BiWeekly => 26, // Annual pattern
            TimeGranularity::Monthly => 12,  // Annual pattern
            TimeGranularity::Quarterly => 4, // Annual pattern
            TimeGranularity::Annual => 1,
        }
    }
}

/// Type of seasonality detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum SeasonalityType {
    /// No significant seasonality
    None = 0,
    /// Weekly pattern (e.g., payroll every Friday)
    Weekly = 1,
    /// Bi-weekly pattern
    BiWeekly = 2,
    /// Monthly pattern (e.g., rent, subscriptions)
    Monthly = 3,
    /// Quarterly pattern (e.g., tax payments, dividends)
    Quarterly = 4,
    /// Semi-annual pattern
    SemiAnnual = 5,
    /// Annual pattern (e.g., year-end adjustments)
    Annual = 6,
}

impl SeasonalityType {
    /// Period length for this seasonality (in days).
    pub fn period_days(&self) -> u32 {
        match self {
            SeasonalityType::None => 0,
            SeasonalityType::Weekly => 7,
            SeasonalityType::BiWeekly => 14,
            SeasonalityType::Monthly => 30,
            SeasonalityType::Quarterly => 91,
            SeasonalityType::SemiAnnual => 182,
            SeasonalityType::Annual => 365,
        }
    }

    /// Common business examples.
    pub fn examples(&self) -> &'static str {
        match self {
            SeasonalityType::None => "No regular pattern",
            SeasonalityType::Weekly => "Payroll, weekly invoicing",
            SeasonalityType::BiWeekly => "Bi-weekly payroll",
            SeasonalityType::Monthly => "Rent, subscriptions, utility payments",
            SeasonalityType::Quarterly => "Tax payments, dividends, quarterly filings",
            SeasonalityType::SemiAnnual => "Insurance premiums, bond interest",
            SeasonalityType::Annual => "Year-end adjustments, depreciation",
        }
    }
}

/// Detected seasonal pattern for an account.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct SeasonalPattern {
    /// Pattern identifier
    pub id: Uuid,
    /// Account this pattern belongs to
    pub account_id: u16,
    /// Type of seasonality
    pub seasonality_type: SeasonalityType,
    /// Period length (in time units)
    pub period_length: u16,
    /// Number of complete cycles observed
    pub cycle_count: u16,
    /// Padding
    pub _pad: u16,
    /// Confidence in the pattern (0.0 - 1.0)
    pub confidence: f32,
    /// Peak autocorrelation value
    pub autocorrelation_peak: f32,
    /// Amplitude of seasonal component
    pub seasonal_amplitude: f32,
    /// Baseline (deseasonalized mean)
    pub baseline: f32,
    /// Trend coefficient (positive = increasing)
    pub trend_coefficient: f32,
    /// Variance in residuals
    pub residual_variance: f32,
    /// First observation
    pub first_observed: HybridTimestamp,
    /// Last observation
    pub last_observed: HybridTimestamp,
    /// Flags
    pub flags: SeasonalPatternFlags,
}

/// Flags for seasonal patterns.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SeasonalPatternFlags(pub u32);

impl SeasonalPatternFlags {
    /// Flag: Pattern is statistically significant.
    pub const IS_STATISTICALLY_SIGNIFICANT: u32 = 1 << 0;
    /// Flag: Pattern shows upward trend.
    pub const HAS_UPWARD_TREND: u32 = 1 << 1;
    /// Flag: Pattern shows downward trend.
    pub const HAS_DOWNWARD_TREND: u32 = 1 << 2;
    /// Flag: Pattern is stable over time.
    pub const IS_STABLE: u32 = 1 << 3;
    /// Flag: Structural break detected in pattern.
    pub const HAS_STRUCTURAL_BREAK: u32 = 1 << 4;
}

impl SeasonalPattern {
    /// Create a new seasonal pattern for an account.
    pub fn new(account_id: u16, seasonality_type: SeasonalityType) -> Self {
        Self {
            id: Uuid::new_v4(),
            account_id,
            seasonality_type,
            period_length: seasonality_type.period_days() as u16,
            cycle_count: 0,
            _pad: 0,
            confidence: 0.0,
            autocorrelation_peak: 0.0,
            seasonal_amplitude: 0.0,
            baseline: 0.0,
            trend_coefficient: 0.0,
            residual_variance: 0.0,
            first_observed: HybridTimestamp::zero(),
            last_observed: HybridTimestamp::zero(),
            flags: SeasonalPatternFlags(0),
        }
    }

    /// Check if pattern is strong (confidence > 0.9).
    pub fn is_strong(&self) -> bool {
        self.confidence > 0.9
    }

    /// Check if pattern is statistically significant.
    pub fn is_significant(&self) -> bool {
        self.flags.0 & SeasonalPatternFlags::IS_STATISTICALLY_SIGNIFICANT != 0
    }
}

/// Behavioral baseline for anomaly detection.
/// Captures "normal" patterns for an account.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct BehavioralBaseline {
    /// Baseline identifier
    pub id: Uuid,
    /// Account this baseline belongs to
    pub account_id: u16,
    /// Number of historical periods used
    pub period_count: u16,
    /// Padding
    pub _pad: u32,

    // === Central tendency ===
    /// Arithmetic mean
    pub mean: f64,
    /// Median (50th percentile)
    pub median: f64,

    // === Dispersion ===
    /// Standard deviation
    pub std_dev: f64,
    /// Median Absolute Deviation (robust to outliers)
    pub mad: f64,

    // === Distribution shape ===
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,
    /// Minimum observed value
    pub min_value: f64,
    /// Maximum observed value
    pub max_value: f64,
    /// 5th percentile
    pub p5: f64,
    /// 95th percentile
    pub p95: f64,

    // === Distribution characteristics ===
    /// Skewness (0 = symmetric, >0 = right tail, <0 = left tail)
    pub skewness: f64,
    /// Kurtosis (3 = normal, >3 = heavy tails)
    pub kurtosis: f64,

    // === Activity patterns ===
    /// Average transaction count per period
    pub avg_transaction_count: f64,
    /// Std dev of transaction counts
    pub transaction_count_std_dev: f64,

    // === Timestamps ===
    /// When baseline was last computed
    pub last_updated: HybridTimestamp,
    /// Start of baseline period
    pub period_start: HybridTimestamp,
    /// End of baseline period
    pub period_end: HybridTimestamp,
}

impl BehavioralBaseline {
    /// Create a new behavioral baseline for an account.
    pub fn new(account_id: u16) -> Self {
        Self {
            id: Uuid::new_v4(),
            account_id,
            period_count: 0,
            _pad: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            mad: 0.0,
            q1: 0.0,
            q3: 0.0,
            iqr: 0.0,
            min_value: 0.0,
            max_value: 0.0,
            p5: 0.0,
            p95: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            avg_transaction_count: 0.0,
            transaction_count_std_dev: 0.0,
            last_updated: HybridTimestamp::zero(),
            period_start: HybridTimestamp::zero(),
            period_end: HybridTimestamp::zero(),
        }
    }

    /// Compute z-score for a value.
    pub fn z_score(&self, value: f64) -> f64 {
        if self.std_dev > 0.0 {
            (value - self.mean) / self.std_dev
        } else {
            0.0
        }
    }

    /// Compute modified z-score (robust, uses MAD).
    pub fn modified_z_score(&self, value: f64) -> f64 {
        if self.mad > 0.0 {
            0.6745 * (value - self.median) / self.mad
        } else {
            0.0
        }
    }

    /// Check if value is an IQR outlier.
    pub fn is_iqr_outlier(&self, value: f64) -> bool {
        let lower_fence = self.q1 - 1.5 * self.iqr;
        let upper_fence = self.q3 + 1.5 * self.iqr;
        value < lower_fence || value > upper_fence
    }

    /// Check if value is a percentile outlier.
    pub fn is_percentile_outlier(&self, value: f64) -> bool {
        value < self.p5 || value > self.p95
    }

    /// Multi-method anomaly check.
    pub fn is_anomaly(&self, value: f64) -> (bool, f32) {
        let mut votes = 0;
        let mut max_severity = 0.0f32;

        // Method 1: Z-score > 3
        let z = self.z_score(value).abs();
        if z > 3.0 {
            votes += 1;
            max_severity = max_severity.max((z / 5.0) as f32);
        }

        // Method 2: Modified z-score > 3.5
        let mz = self.modified_z_score(value).abs();
        if mz > 3.5 {
            votes += 1;
            max_severity = max_severity.max((mz / 5.0) as f32);
        }

        // Method 3: IQR outlier
        if self.is_iqr_outlier(value) {
            votes += 1;
            let iqr_deviation = if value < self.q1 {
                (self.q1 - value) / self.iqr
            } else {
                (value - self.q3) / self.iqr
            };
            max_severity = max_severity.max(iqr_deviation as f32);
        }

        // Method 4: Percentile outlier
        if self.is_percentile_outlier(value) {
            votes += 1;
        }

        // Anomaly if >= 2 methods agree
        let is_anomaly = votes >= 2;
        let score = match votes {
            2 => 0.5,
            3 => 0.75,
            4 => 1.0,
            _ => 0.0,
        } * max_severity.min(1.0);

        (is_anomaly, score)
    }
}

/// Time series metrics for trend/volatility analysis.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct TimeSeriesMetrics {
    /// Metrics identifier
    pub id: Uuid,
    /// Account this belongs to
    pub account_id: u16,
    /// Number of periods in the series
    pub period_count: u16,
    /// Time granularity
    pub granularity: TimeGranularity,
    /// Padding
    pub _pad: [u8; 3],

    // === Trend ===
    /// Linear trend slope (OLS)
    pub trend_coefficient: f64,
    /// Trend intercept
    pub trend_intercept: f64,
    /// R-squared for trend fit
    pub trend_r_squared: f64,

    // === Volatility ===
    /// Standard deviation of values
    pub volatility: f64,
    /// Coefficient of variation (std_dev / mean)
    pub coefficient_of_variation: f64,

    // === Moving averages ===
    /// Simple moving average (last period)
    pub sma: f64,
    /// Exponential moving average (last period)
    pub ema: f64,

    // === Current state ===
    /// Most recent value
    pub current_value: f64,
    /// Forecasted next value
    pub forecasted_value: f64,
    /// Forecast confidence interval (95%)
    pub forecast_ci: f64,

    // === Timestamps ===
    /// Start of the time series period.
    pub period_start: HybridTimestamp,

    // === Flags ===
    /// Time series property flags.
    pub flags: TimeSeriesFlags,
}

/// Flags for time series properties.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct TimeSeriesFlags(pub u32);

impl TimeSeriesFlags {
    /// Flag: Series has a statistically significant trend.
    pub const HAS_SIGNIFICANT_TREND: u32 = 1 << 0;
    /// Flag: Series is increasing (upward trend).
    pub const IS_INCREASING: u32 = 1 << 1;
    /// Flag: Series is decreasing (downward trend).
    pub const IS_DECREASING: u32 = 1 << 2;
    /// Flag: Series has high volatility.
    pub const IS_HIGH_VOLATILITY: u32 = 1 << 3;
    /// Flag: Series is stationary (no trend).
    pub const IS_STATIONARY: u32 = 1 << 4;
}

impl TimeSeriesMetrics {
    /// Create new time series metrics for an account.
    pub fn new(account_id: u16, granularity: TimeGranularity) -> Self {
        Self {
            id: Uuid::new_v4(),
            account_id,
            period_count: 0,
            granularity,
            _pad: [0; 3],
            trend_coefficient: 0.0,
            trend_intercept: 0.0,
            trend_r_squared: 0.0,
            volatility: 0.0,
            coefficient_of_variation: 0.0,
            sma: 0.0,
            ema: 0.0,
            current_value: 0.0,
            forecasted_value: 0.0,
            forecast_ci: 0.0,
            period_start: HybridTimestamp::zero(),
            flags: TimeSeriesFlags(0),
        }
    }

    /// Check if there's a significant upward trend.
    pub fn is_increasing(&self) -> bool {
        self.flags.0 & TimeSeriesFlags::IS_INCREASING != 0
    }

    /// Check if there's a significant downward trend.
    pub fn is_decreasing(&self) -> bool {
        self.flags.0 & TimeSeriesFlags::IS_DECREASING != 0
    }

    /// Check if volatility is high (CV > 0.5).
    pub fn is_volatile(&self) -> bool {
        self.flags.0 & TimeSeriesFlags::IS_HIGH_VOLATILITY != 0
    }
}

/// An alert generated by temporal analysis.
#[derive(Debug, Clone)]
pub struct TemporalAlert {
    /// Alert identifier
    pub id: Uuid,
    /// Account involved
    pub account_id: u16,
    /// Alert type
    pub alert_type: TemporalAlertType,
    /// Severity (0.0 - 1.0)
    pub severity: f32,
    /// Value that triggered the alert
    pub trigger_value: f64,
    /// Expected value (baseline)
    pub expected_value: f64,
    /// Deviation (trigger - expected)
    pub deviation: f64,
    /// When the alert was raised
    pub timestamp: HybridTimestamp,
    /// Human-readable message
    pub message: String,
}

/// Types of temporal alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalAlertType {
    /// Value outside normal range
    Anomaly,
    /// Sudden trend change
    TrendBreak,
    /// Missing expected seasonal activity
    SeasonalDeviation,
    /// Activity after long dormancy
    DormantActivation,
    /// Rapid value changes
    HighVolatility,
    /// Unusual transaction frequency
    FrequencyAnomaly,
}

impl TemporalAlertType {
    /// Get icon character for this alert type.
    pub fn icon(&self) -> &'static str {
        match self {
            TemporalAlertType::Anomaly => "⚠️",
            TemporalAlertType::TrendBreak => "📈",
            TemporalAlertType::SeasonalDeviation => "📅",
            TemporalAlertType::DormantActivation => "💤",
            TemporalAlertType::HighVolatility => "📊",
            TemporalAlertType::FrequencyAnomaly => "🔢",
        }
    }

    /// Get description of this alert type.
    pub fn description(&self) -> &'static str {
        match self {
            TemporalAlertType::Anomaly => "Value outside normal range",
            TemporalAlertType::TrendBreak => "Significant trend change detected",
            TemporalAlertType::SeasonalDeviation => "Deviation from seasonal pattern",
            TemporalAlertType::DormantActivation => "Activity on dormant account",
            TemporalAlertType::HighVolatility => "Unusually high value volatility",
            TemporalAlertType::FrequencyAnomaly => "Unusual transaction frequency",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_anomaly_detection() {
        let mut baseline = BehavioralBaseline::new(0);
        baseline.mean = 100.0;
        baseline.std_dev = 10.0;
        baseline.median = 100.0;
        baseline.mad = 7.5;
        baseline.q1 = 90.0;
        baseline.q3 = 110.0;
        baseline.iqr = 20.0;
        baseline.p5 = 80.0;
        baseline.p95 = 120.0;

        // Normal value
        let (is_anomaly, _) = baseline.is_anomaly(105.0);
        assert!(!is_anomaly);

        // Clear outlier
        let (is_anomaly, score) = baseline.is_anomaly(200.0);
        assert!(is_anomaly);
        assert!(score > 0.5);
    }

    #[test]
    fn test_z_score() {
        let mut baseline = BehavioralBaseline::new(0);
        baseline.mean = 100.0;
        baseline.std_dev = 10.0;

        let z = baseline.z_score(130.0);
        assert!((z - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_seasonality_period() {
        assert_eq!(SeasonalityType::Monthly.period_days(), 30);
        assert_eq!(SeasonalityType::Quarterly.period_days(), 91);
        assert_eq!(SeasonalityType::Annual.period_days(), 365);
    }
}
