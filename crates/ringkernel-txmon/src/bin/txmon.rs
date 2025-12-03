//! Transaction Monitoring GUI Application
//!
//! Run with: `cargo run -p ringkernel-txmon --bin txmon`

use iced::{Size, Task};
use ringkernel_txmon::gui::TxMonApp;

fn main() -> iced::Result {
    tracing_subscriber::fmt()
        .with_env_filter("ringkernel_txmon=debug,warn")
        .init();

    iced::application("RingKernel Transaction Monitor", TxMonApp::update, TxMonApp::view)
        .subscription(TxMonApp::subscription)
        .theme(TxMonApp::theme)
        .window_size(Size::new(1400.0, 900.0))
        .run_with(|| (TxMonApp::new(), Task::none()))
}
