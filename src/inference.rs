use std::{fmt::{Debug, Display}, str::FromStr};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::{model::Pipeline, prelude::{ModelType, TokenString}};

pub struct InferIter {
    model_type: ModelType,
    device: Device,
    tokens: TokenString,
    vocab_size: usize,
    step: usize,
    pipeline: Pipeline,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
    reached_eos: bool,
}

impl InferIter {
    pub(crate) fn new(
        model_type: ModelType,
        device: Device,
        tokens: TokenString,
        vocab_size: usize,
        pipeline: Pipeline,
        logits_processor: LogitsProcessor,
        repeat_penalty: f32,
        repeat_last_n: usize,
        eos_token: u32,
    ) -> Self {
        Self {
            model_type,
            device,
            tokens,
            vocab_size,
            step: 0,
            pipeline,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            eos_token,
            reached_eos: false,
        }
    }

    pub fn next_token(&mut self) -> Option<u32> {
        // Exit early if we already got the end of text token
        if self.reached_eos {
            return None;
        }

        // Get the context size for this step
        let context_size = if self.step > 0 { 1 } else { self.tokens.len() };
        
        // Get the start position for the context
        let start_pos = self.tokens.len().saturating_sub(context_size);

        // Get the context
        let context = self
            .tokens
            .get(start_pos..)
            .unwrap();

        // Create the input tensor containing the context
        let input = Tensor::new(context, &self.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // Forward the input through the pipeline
        let logits = self.pipeline.forward(&input, start_pos);

        // Preprocess the logits for this model type
        let logits = self.model_type.process_logits(logits);

        // Apply the repeat penalty
        let logits = if self.repeat_penalty == 1.0 || self.repeat_last_n == 0 {
            logits
        } else {
            // Apply the repeat penalty to the last repeat_last_n tokens
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                self.tokens.get(start_at..).unwrap(),
            )
            .unwrap()
        };

        // Sample the next token
        let next_token = self.logits_processor.sample(&logits).unwrap();

        // Increment the step
        self.step += 1;

        // If the token is not the end of text token, add it to the tokens
        if next_token != self.eos_token {
            self.tokens.push_token(next_token);
        }
        // Otherwise, set reached_eos to true and return None
        else {
            self.reached_eos = true;
            return None;
        }

        // Return the next token
        Some(next_token)
    }

    /// Run the iterator until completion or until one of `end_sequences` is generated
    /// and return everything up to that point as a `String`, as well as the end sequence that was reached
    pub fn complete<'a>(mut self, end_sequences: &'a [&str]) -> (String, Option<&'a str>) {
        let mut response = String::new();
        while let Some(token) = self.next_token()
            && token < self.vocab_size as u32 - 1
        {
            let token_str = self.tokens.model.detokenize(&[token]);

            response.push_str(&token_str);

            // Exit early at the first stop sequence from end_sequences encountered in response, truncating.
            // Only search in the last END_SEQUENCE_SEARCH_WINDOW characters of the response
            let found_stop_sequence_position = end_sequences
                .iter()
                .enumerate()
                .filter_map(|(idx, &seq)| response.find(seq).map(|pos| (idx, pos)))
                .min_by_key(|&(_, pos)| pos);

            if let Some((idx, pos)) = found_stop_sequence_position {
                response.truncate(pos);
                self.reached_eos = true;
                return (response, Some(end_sequences[idx]));
            }
        }

        self.reached_eos = true;

        (response, None)
    }
}

impl Into<String> for InferIter {
    fn into(self) -> String {
        self.complete(&[]).0
    }
}

impl Iterator for InferIter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}


#[derive(Clone)]
pub enum InferValue {
    String(String),
    Float(f64),
    Int(i64),
}

impl InferValue {
    /// Convert the `InferValue` to a simple `String`
    pub fn to_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Float(f) => f.to_string(),
            Self::Int(i) => i.to_string(),
        }
    }

    /// Parse a `&str` to an `InferValue`
    pub fn from_str(s: &str) -> Result<Self> {
        if let Ok(f) = s.parse::<f64>() {
            Ok(Self::Float(f))
        } else if let Ok(i) = s.parse::<i64>() {
            Ok(Self::Int(i))
        } else {
            Ok(Self::String(s.to_string().trim_matches('"').to_string()))
        }
    }

    /// Write a debug representation of the `InferValue`.
    /// This is the same as the value would appear in code.
    pub fn write_debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "\"{}\"", s),
            Self::Float(fl) => write!(f, "{}", fl),
            Self::Int(i) => write!(f, "{}", i),
        }
    }
}

impl FromStr for InferValue {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        InferValue::from_str(s)
    }
}

impl Display for InferValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Debug for InferValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_debug(f)
    }
}

impl From<String> for InferValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for InferValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_owned())
    }
}

impl From<f64> for InferValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<i64> for InferValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<f32> for InferValue {
    fn from(f: f32) -> Self {
        Self::Float(f as f64)
    }
}

impl From<i32> for InferValue {
    fn from(i: i32) -> Self {
        Self::Int(i as i64)
    }
}

impl From<u32> for InferValue {
    fn from(i: u32) -> Self {
        Self::Int(i as i64)
    }
}

impl From<i16> for InferValue {
    fn from(i: i16) -> Self {
        Self::Int(i as i64)
    }
}

impl From<u16> for InferValue {
    fn from(i: u16) -> Self {
        Self::Int(i as i64)
    }
}

impl From<i8> for InferValue {
    fn from(i: i8) -> Self {
        Self::Int(i as i64)
    }
}

impl From<u8> for InferValue {
    fn from(i: u8) -> Self {
        Self::Int(i as i64)
    }
}

impl From<usize> for InferValue {
    fn from(i: usize) -> Self {
        Self::Int(i as i64)
    }
}

impl From<isize> for InferValue {
    fn from(i: isize) -> Self {
        Self::Int(i as i64)
    }
}

impl<T: Into<InferValue> + Clone> From<&T> for InferValue {
    fn from(value: &T) -> Self {
        value.clone().into()
    }
}

impl Into<String> for InferValue {
    fn into(self) -> String {
        match self {
            InferValue::String(s) => {
                // Remove surrounding quotes (both ' and ")
                s.trim_matches(|c| c == '"' || c == '\'').to_string()
            }
            InferValue::Float(f) => f.to_string(),
            InferValue::Int(i) => i.to_string(),
        }
    }
}

impl TryInto<f64> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<f64, Self::Error> {
        match self {
            InferValue::Float(f) => Ok(f),
            InferValue::Int(i) => Ok(i as f64),
            InferValue::String(s) => s.parse::<f64>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<i64> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            InferValue::Int(i) => Ok(i),
            InferValue::Float(f) => Ok(f as i64),
            InferValue::String(s) => s.parse::<i64>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<u64> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u64, Self::Error> {
        match self {
            InferValue::Int(i) => Ok(i as u64),
            InferValue::Float(f) => Ok(f as u64),
            InferValue::String(s) => s.parse::<u64>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<f32> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<f32, Self::Error> {
        match self {
            InferValue::Float(f) => Ok(f as f32),
            InferValue::Int(i) => Ok(i as f32),
            InferValue::String(s) => s.parse::<f32>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<i32> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i32, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to i32")),
            InferValue::Float(f) => Ok(f as i32),
            InferValue::String(s) => s.parse::<i32>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<u32> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u32, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to u32")),
            InferValue::Float(f) => Ok(f as u32),
            InferValue::String(s) => s.parse::<u32>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<i16> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i16, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to i16")),
            InferValue::Float(f) => Ok(f as i16),
            InferValue::String(s) => s.parse::<i16>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<u16> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u16, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to u16")),
            InferValue::Float(f) => Ok(f as u16),
            InferValue::String(s) => s.parse::<u16>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<i8> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<i8, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to i8")),
            InferValue::Float(f) => Ok(f as i8),
            InferValue::String(s) => s.parse::<i8>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<u8> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<u8, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to u8")),
            InferValue::Float(f) => Ok(f as u8),
            InferValue::String(s) => s.parse::<u8>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<isize> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<isize, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to isize")),
            InferValue::Float(f) => Ok(f as isize),
            InferValue::String(s) => s.parse::<isize>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

impl TryInto<usize> for InferValue {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<usize, Self::Error> {
        match self {
            InferValue::Int(i) => i
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to convert i64 to usize")),
            InferValue::Float(f) => Ok(f as usize),
            InferValue::String(s) => s.parse::<usize>().map_err(|e| anyhow::anyhow!(e)),
        }
    }
}

#[macro_export]
macro_rules! data_map {
    [$($key:expr => $value:expr),*$(,)?] => {
        {
            #[allow(unused_mut)]
            let mut map: std::collections::HashMap<String, $crate::model::InferValue> = std::collections::HashMap::new();
            $(
                map.insert($key.to_string(), $crate::model::InferValue::from($value));
            )*
            map
        }
    };
}