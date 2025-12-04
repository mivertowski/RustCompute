//! Token animation for case flow visualization.
//!
//! Animates tokens flowing along DFG edges.

/// Active token representing a case flowing through the process.
#[derive(Debug, Clone)]
pub struct Token {
    /// Token ID.
    pub id: u64,
    /// Case ID this token represents.
    pub case_id: u64,
    /// Source activity.
    pub source_activity: u32,
    /// Target activity.
    pub target_activity: u32,
    /// Progress along edge (0.0 to 1.0).
    pub progress: f32,
    /// Speed multiplier.
    pub speed: f32,
}

impl Token {
    /// Create a new token.
    pub fn new(id: u64, case_id: u64, source: u32, target: u32) -> Self {
        Self {
            id,
            case_id,
            source_activity: source,
            target_activity: target,
            progress: 0.0,
            speed: 1.0,
        }
    }

    /// Update token position.
    pub fn update(&mut self, dt: f32) -> bool {
        self.progress += dt * self.speed;
        self.progress >= 1.0
    }
}

/// Token animation manager.
#[derive(Debug, Default)]
pub struct TokenAnimation {
    /// Active tokens.
    tokens: Vec<Token>,
    /// Next token ID.
    next_id: u64,
    /// Maximum concurrent tokens.
    max_tokens: usize,
    /// Base animation speed.
    base_speed: f32,
}

impl TokenAnimation {
    /// Create a new token animation manager.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            next_id: 1,
            max_tokens: 50,
            base_speed: 0.5,
        }
    }

    /// Set maximum tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set base speed.
    pub fn with_base_speed(mut self, speed: f32) -> Self {
        self.base_speed = speed;
        self
    }

    /// Spawn a new token.
    pub fn spawn(&mut self, case_id: u64, source: u32, target: u32) {
        if self.tokens.len() >= self.max_tokens {
            // Remove oldest token
            self.tokens.remove(0);
        }

        let mut token = Token::new(self.next_id, case_id, source, target);
        token.speed = self.base_speed;
        self.tokens.push(token);
        self.next_id += 1;
    }

    /// Spawn multiple tokens from events.
    pub fn spawn_from_transitions(&mut self, transitions: &[(u64, u32, u32)]) {
        for &(case_id, source, target) in transitions {
            self.spawn(case_id, source, target);
        }
    }

    /// Update all tokens.
    pub fn update(&mut self, dt: f32) {
        // Update and remove completed tokens
        self.tokens.retain_mut(|token| !token.update(dt));
    }

    /// Get active tokens.
    pub fn active_tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Get token count.
    pub fn count(&self) -> usize {
        self.tokens.len()
    }

    /// Clear all tokens.
    pub fn clear(&mut self) {
        self.tokens.clear();
    }

    /// Set animation speed.
    pub fn set_speed(&mut self, speed: f32) {
        self.base_speed = speed;
        for token in &mut self.tokens {
            token.speed = speed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token::new(1, 100, 1, 2);
        assert_eq!(token.progress, 0.0);
        assert_eq!(token.source_activity, 1);
        assert_eq!(token.target_activity, 2);
    }

    #[test]
    fn test_token_update() {
        let mut token = Token::new(1, 100, 1, 2);
        token.speed = 1.0;

        // Update to 50%
        let completed = token.update(0.5);
        assert!(!completed);
        assert_eq!(token.progress, 0.5);

        // Update to 100%
        let completed = token.update(0.5);
        assert!(completed);
    }

    #[test]
    fn test_animation_manager() {
        let mut anim = TokenAnimation::new().with_max_tokens(5);
        anim.spawn(100, 1, 2);
        anim.spawn(101, 2, 3);

        assert_eq!(anim.count(), 2);

        // Update
        anim.update(0.1);
        assert_eq!(anim.count(), 2);

        // Update until completion
        for _ in 0..20 {
            anim.update(0.1);
        }
        assert_eq!(anim.count(), 0);
    }
}
