//! Transaction data types.

use bytemuck::{Pod, Zeroable};

/// Transaction type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum TransactionType {
    /// Wire transfer.
    #[default]
    Wire = 0,
    /// ACH transfer.
    ACH = 1,
    /// Check deposit/payment.
    Check = 2,
    /// Card transaction.
    Card = 3,
    /// Cash transaction.
    Cash = 4,
    /// Internal transfer between accounts.
    Internal = 5,
}

impl TransactionType {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            TransactionType::Wire => "Wire",
            TransactionType::ACH => "ACH",
            TransactionType::Check => "Check",
            TransactionType::Card => "Card",
            TransactionType::Cash => "Cash",
            TransactionType::Internal => "Internal",
        }
    }

    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(TransactionType::Wire),
            1 => Some(TransactionType::ACH),
            2 => Some(TransactionType::Check),
            3 => Some(TransactionType::Card),
            4 => Some(TransactionType::Cash),
            5 => Some(TransactionType::Internal),
            _ => None,
        }
    }
}

impl std::fmt::Display for TransactionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// 128-byte GPU-aligned transaction structure.
///
/// Contains all information about a financial transaction for monitoring.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(128))]
pub struct Transaction {
    /// Unique transaction identifier.
    pub transaction_id: u64,
    /// Source customer identifier.
    pub customer_id: u64,
    /// Destination customer identifier (0 for external).
    pub destination_id: u64,
    /// Transaction amount in cents.
    pub amount_cents: u64,
    /// Destination country code (ISO 3166-1 numeric).
    pub country_code: u16,
    /// Transaction type (cast to TransactionType).
    pub tx_type: u8,
    /// Currency code (ISO 4217 numeric, e.g., 840 = USD).
    pub currency: u8,
    /// Flags (bitmask for various attributes).
    pub flags: u32,
    /// Timestamp (Unix epoch milliseconds).
    pub timestamp: u64,
    /// Source account hash (for pattern detection).
    pub source_hash: u32,
    /// Destination account hash.
    pub dest_hash: u32,
    /// Reference number (for linking related transactions).
    pub reference: u64,
    /// Reserved for future use.
    _reserved: [u8; 64],
}

const _: () = assert!(std::mem::size_of::<Transaction>() == 128);

impl Transaction {
    /// Create a new transaction.
    pub fn new(
        transaction_id: u64,
        customer_id: u64,
        amount_cents: u64,
        country_code: u16,
        timestamp: u64,
    ) -> Self {
        Self {
            transaction_id,
            customer_id,
            destination_id: 0,
            amount_cents,
            country_code,
            tx_type: TransactionType::Wire as u8,
            currency: 184, // 840 = USD, but we use u8 so we'll use a smaller code
            flags: 0,
            timestamp,
            source_hash: 0,
            dest_hash: 0,
            reference: 0,
            _reserved: [0; 64],
        }
    }

    /// Get the transaction type enum.
    pub fn tx_type(&self) -> TransactionType {
        TransactionType::from_u8(self.tx_type).unwrap_or_default()
    }

    /// Format amount as currency string.
    pub fn format_amount(&self) -> String {
        let dollars = self.amount_cents / 100;
        let cents = self.amount_cents % 100;
        format!("${}.{:02}", dollars, cents)
    }

    /// Check if this is a high-value transaction (>$10,000).
    pub fn is_high_value(&self) -> bool {
        self.amount_cents >= 1_000_000 // $10,000 in cents
    }

    /// Check if this is an international transaction.
    pub fn is_international(&self, home_country: u16) -> bool {
        self.country_code != home_country
    }

    /// Get country code as string (simplified mapping).
    pub fn country_name(&self) -> &'static str {
        match self.country_code {
            1 => "US",
            2 => "UK",
            3 => "DE",
            4 => "FR",
            5 => "JP",
            6 => "CN",
            7 => "RU",
            8 => "BR",
            9 => "IN",
            10 => "CA",
            11 => "AU",
            12 => "MX",
            13 => "KR",
            14 => "IT",
            15 => "ES",
            _ => "??",
        }
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_size() {
        assert_eq!(std::mem::size_of::<Transaction>(), 128);
    }

    #[test]
    fn test_format_amount() {
        let tx = Transaction::new(1, 1, 123_456, 1, 0);
        assert_eq!(tx.format_amount(), "$1234.56");
    }

    #[test]
    fn test_high_value() {
        let low = Transaction::new(1, 1, 50_000, 1, 0);
        let high = Transaction::new(2, 1, 1_500_000, 1, 0);

        assert!(!low.is_high_value());
        assert!(high.is_high_value());
    }
}
