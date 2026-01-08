//! Error types for the RingKernel CLI.

use thiserror::Error;

/// CLI result type alias.
pub type CliResult<T> = Result<T, CliError>;

/// CLI error type.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum CliError {
    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Template rendering error.
    #[error("Template error: {0}")]
    Template(String),

    /// Invalid project name.
    #[error("Invalid project name: {0}")]
    InvalidProjectName(String),

    /// Template not found.
    #[error("Template not found: {0}")]
    TemplateNotFound(String),

    /// Invalid backend specification.
    #[error("Invalid backend: {0}")]
    InvalidBackend(String),

    /// Code generation error.
    #[error("Code generation failed: {0}")]
    CodegenError(String),

    /// Parse error when reading source files.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Project already exists.
    #[error("Project already exists at: {0}")]
    ProjectExists(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Validation error.
    #[error("Validation failed: {0}")]
    Validation(String),

    /// Feature not available.
    #[error("Feature not available: {0}. Enable with --features {1}")]
    FeatureNotAvailable(String, String),
}

impl From<handlebars::RenderError> for CliError {
    fn from(e: handlebars::RenderError) -> Self {
        CliError::Template(e.to_string())
    }
}

impl From<handlebars::TemplateError> for CliError {
    fn from(e: handlebars::TemplateError) -> Self {
        CliError::Template(e.to_string())
    }
}

impl From<toml::de::Error> for CliError {
    fn from(e: toml::de::Error) -> Self {
        CliError::Config(e.to_string())
    }
}

impl From<toml::ser::Error> for CliError {
    fn from(e: toml::ser::Error) -> Self {
        CliError::Config(e.to_string())
    }
}

impl From<syn::Error> for CliError {
    fn from(e: syn::Error) -> Self {
        CliError::ParseError(e.to_string())
    }
}
