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

    /// Push any type that can be converted into a token string
    pub fn push(&mut self, other: impl IntoTokenString) {
        self.tokens
            .extend(other.into_token_string(&self.model).tokens);
    }

    /// Push a single token
    pub fn push_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Push many tokens
    pub fn push_tokens(&mut self, tokens: impl AsRef<[u32]>) {
        self.tokens.extend(tokens.as_ref());
    }

    /// Push many tokens from an iterator
    pub fn extend(&mut self, tokens: impl IntoIterator<Item = u32>) {
        self.tokens.extend(tokens);
    }

    /// Encode a type that implements `Display` into tokens and push them
    pub fn push_str(&mut self, text: impl Display) {
        self.extend(self.model.tokenize_str(text));
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

/// A trait for types that can be converted into a `TokenString`
pub trait IntoTokenString {
    /// Convert the value into a `TokenString` using the provided model
    fn into_token_string(self, model: &Model) -> TokenString;
}

impl IntoTokenString for &str {
    fn into_token_string(self, model: &Model) -> TokenString {
        model.tokenize_str(self)
    }
}

impl IntoTokenString for String {
    fn into_token_string(self, model: &Model) -> TokenString {
        model.tokenize_str(self)
    }
}

impl IntoTokenString for &String {
    fn into_token_string(self, model: &Model) -> TokenString {
        model.tokenize_str(self)
    }
}

impl IntoTokenString for TokenString {
    // TODO: this may break if multiple different models are supported later
    fn into_token_string(self, _: &Model) -> TokenString {
        self
    }
}

impl IntoTokenString for &TokenString {
    // TODO: this may break if multiple different models are supported later
    fn into_token_string(self, _: &Model) -> TokenString {
        self.clone()
    }
}
