use std::{fmt::Display, slice::SliceIndex};

use crate::model::Model;

/// A string of tokens representing a sequence of text
#[derive(Clone)]
pub struct TokenString {
    pub tokens: Vec<u32>,
    pub model: Model,
}

impl TokenString {
    /// Create a new TokenString from a list of tokens and a model
    pub(crate) fn new(tokens: Vec<u32>, model: Model) -> Self {
        Self { tokens, model }
    }

    /// Push a token
    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Push many tokens
    pub fn push_many(&mut self, tokens: impl AsRef<[u32]>) {
        self.tokens.extend(tokens.as_ref());
    }

    /// Push many tokens from an iterator
    pub fn extend(&mut self, tokens: impl IntoIterator<Item = u32>) {
        self.tokens.extend(tokens);
    }

    /// Encode a string and push the tokens
    pub fn push_str(&mut self, text: impl Display) {
        self.extend(self.model.tokenize(text));
    }

    /// Truncate the token string to a maximum number of tokens
    pub fn truncate(&mut self, len: usize) {
        self.tokens.truncate(len);
    }

    /// Get the number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if the token string is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get a token or range of tokens by index
    pub(crate) fn get<I: SliceIndex<[u32]>>(&self, index: I) -> Option<&I::Output> {
        self.tokens.get(index)
    }

    /// Get the tokens as a slice
    pub fn as_slice(&self) -> &[u32] {
        &self.tokens
    }

    /// Get the tokens as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.tokens
    }

    /// Get an iterator over the tokens
    pub fn iter(&self) -> std::slice::Iter<'_, u32> {
        self.tokens.iter()
    }

    /// Get the tokens as a vector
    pub fn into_vec(self) -> Vec<u32> {
        self.tokens
    }

    /// Decode the tokens into a new `String`
    pub fn to_string(&self) -> String {
        self.model.detokenize(&self.tokens)
    }

    /// Infer the next tokens using the model
    pub fn next(&self, max_tokens: usize, temp: Option<f64>, stop_at: &[&str]) -> TokenString {
        // Generate the seed from the last 4 tokens
        let seed = self
            .get(self.len().saturating_sub(4)..)
            .unwrap()
            .iter()
            .fold(0u64, |acc, &token| acc.wrapping_add(token as u64));

        self.next_2(seed, max_tokens, temp, None, 0.0, 0, stop_at)
    }

    /// Infer the next tokens using the model, with additional parameters
    pub fn next_2(
        &self,
        seed: u64,
        max_tokens: usize,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        stop_at: &[&str],
    ) -> TokenString {
        // Begin inference
        let infer_iter = self.model.infer_iter(
            self.clone(),
            seed,
            temp,
            top_p,
            repeat_penalty,
            repeat_last_n,
        ).unwrap();

        // Collect the tokens until a stopping token is reached
        let mut tokens = self.model.new_token_string();
        for token in infer_iter {
            // Push the token
            tokens.push(token);

            // Detokenize the token
            let token_str = self.model.detokenize(&[token]);

            // Check if the token string ends with a stopping token
            if stop_at.iter().any(|&stop| token_str.ends_with(stop)) {
                break;
            }

            // Check if the maximum number of tokens has been reached
            if tokens.len() >= max_tokens {
                break;
            }
        }

        // Return the token string
        tokens
    }

    /// Infer the next tokens using the model, and return the complete token string
    pub fn completed(&self, max_new_tokens: usize, stop_at: &[&str]) -> TokenString {
        // Clone self and append the next tokens
        let mut completed = self.clone();
        completed.push_many(self.next(max_new_tokens, Some(1.0), stop_at));

        completed
    }

    /// Infer the next tokens using the model, and return the complete token string,
    /// with additional parameters
    pub fn completed_2(
        &self,
        seed: u64,
        max_new_tokens: usize,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        stop_at: &[&str],
    ) -> TokenString {
        // Clone self and append the next tokens
        let mut completed = self.clone();
        completed.push_many(self.next_2(
            seed,
            max_new_tokens,
            temp,
            top_p,
            repeat_penalty,
            repeat_last_n,
            stop_at,
        ));

        completed
    }

    /// Submit the tokens as an instruction and retrieve the response
    pub fn instruct(&self, stop_at: &[&str]) -> TokenString {
        // Perform the task
        self.model.instruct(self, stop_at).unwrap()
    }

    /// Submit the tokens as an instruction and retrieve the response, with additional parameters
    pub fn instruct_2(
        &self,
        seed: u64,
        max_tokens: usize,
        temp: Option<f64>,
        stop_at: &[&str],
    ) -> TokenString {
        self.model
            .instruct_2(self, seed, max_tokens, temp, stop_at)
            .unwrap()
    }

    /// Attempt to shorten the token string while retaining important information
    pub fn shortened(&self) -> TokenString {
        // Generate the seed from the first 4 tokens
        let seed = self
            .tokens
            .iter()
            .take(4)
            .fold(0u64, |acc, &x| acc.overflowing_add(x as u64).0);
        // Shorten
        self.shortened_2(seed, self.model.max_tokens())
    }

    /// Attempt to shorten the token string while retaining important information
    pub fn shortened_2(&self, seed: u64, max_tokens: usize) -> TokenString {
        // Generate the instruction
        let mut instruction = self.model.tokenize("Paraphrase the following text:\n");
        instruction.push_many(&self.tokens);
        // Perform the instruction
        self.model
            .instruct_2(&instruction, seed, max_tokens, Some(0.5), &[])
            .unwrap()
    }

    /// Attempts shortening the token string while retaining important information.
    /// Returns the shortened token string if successful, otherwise `None`.
    pub fn shortened_to(&self, max_tokens: usize, max_attempts: usize) -> Option<TokenString> {
        // Generate the seed from the first 4 tokens
        let seed = self
            .tokens
            .iter()
            .take(4)
            .fold(0u64, |acc, &x| acc.overflowing_add(x as u64).0);
        // Generate the instruction
        let mut instruction = self.model.tokenize("Paraphrase the following text:\n");
        instruction.push_many(&self.tokens);
        // Attempt to shorten the token string, incrementing the seed each time
        for i in 0..max_attempts {
            let shortened = self
                .model
                .instruct_2(
                    &instruction,
                    seed.overflowing_add(i as u64).0,
                    max_tokens,
                    Some(0.5),
                    &[],
                )
                .unwrap();
            if shortened.len() <= max_tokens {
                return Some(shortened);
            }
        }
        None
    }
}

impl Into<Vec<u32>> for TokenString {
    fn into(self) -> Vec<u32> {
        self.tokens
    }
}

impl Into<String> for TokenString {
    fn into(self) -> String {
        self.to_string()
    }
}

impl IntoIterator for TokenString {
    type Item = u32;
    type IntoIter = std::vec::IntoIter<u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.tokens.into_iter()
    }
}

impl Display for TokenString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl AsRef<[u32]> for TokenString {
    fn as_ref(&self) -> &[u32] {
        &self.tokens
    }
}

impl AsMut<[u32]> for TokenString {
    fn as_mut(&mut self) -> &mut [u32] {
        &mut self.tokens
    }
}
