use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::str::FromStr;

use anyhow::{anyhow, Error as E, Result};

use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use tokenizers::{Token, Tokenizer};

use crate::scene::Scene;
use crate::token_string::TokenString;

pub const MAX_TOKENS: usize = 2048;

#[derive(Clone)]
pub struct Model {
    config: Config,
    vb: VarBuilder<'static>,
    tokenizer: Tokenizer,
    device: Device,
    seed: u64,
}

impl Model {
    pub fn new(seed: u64, use_cuda: bool) -> Result<Self> {
        let device = if use_cuda && candle_core::utils::cuda_is_available() {
            Device::new_cuda(0).unwrap()
        } else {
            Device::Cpu
        };
        let api = Api::new()?;
        let repo = api.model("lmz/candle-quantized-phi".to_string());
        let tokenizer_filename = repo.get("tokenizer-puffin-phi-v2.json")?;
        let model_filename = repo.get("model-phi-hermes-1_3B.safetensors")?;

        // Create model config
        let config = Config::phi_hermes_1_3b();

        // Create VarBuilder
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };

        // Create tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        Ok(Self {
            config,
            vb,
            tokenizer,
            device: device.clone(),
            seed,
        })
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn max_tokens(&self) -> usize {
        MAX_TOKENS
    }

    pub fn new_token_string(&self) -> TokenString {
        TokenString::new(Vec::new(), self.clone())
    }

    pub fn tokenize(&self, text: impl Display) -> TokenString {
        // Tokenize the text
        let tokens = self.tokenizer.encode(text.to_string(), true).unwrap();

        // Get the token ids
        let token_ids = tokens.get_ids().to_vec();
        TokenString::new(token_ids, self.clone())
    }

    pub(crate) fn detokenize(&self, tokens: &[u32]) -> String {
        // Decode the tokens into a string
        let text = self.tokenizer.decode(tokens, true).map_err(E::msg).unwrap();
        text
    }

    pub fn create_scene(
        &self,
        setting: impl Display,
        starting_characters: &[impl Display],
    ) -> Scene {
        Scene::new(self.clone(), setting, starting_characters)
    }

    /// Return an iterator that can be used to perform inference
    pub(crate) fn infer_iter(
        &self,
        prompt_tokens: TokenString,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<InferIter> {
        // Add the model seed to the seed provided
        let seed = seed.wrapping_add(self.seed);

        // Fail if the prompt is empty
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt was empty")
        }

        // Create pipeline
        let pipeline = MixFormer::new(&self.config, self.vb.clone())?;

        // Create logits processor
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        // Get the end of text token
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };

        // Create the iterator
        Ok(InferIter::new(
            self.device.clone(),
            prompt_tokens,
            pipeline,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            eos_token,
        ))
    }

    pub(crate) fn instruct(
        &self,
        instruction: &TokenString,
        stop_at: &[&str],
    ) -> Result<TokenString> {
        // Generate the seed from the last 4 tokens
        let seed = instruction
            .get(instruction.len().saturating_sub(4)..)
            .unwrap()
            .iter()
            .fold(0u64, |acc, &token| acc.wrapping_add(token as u64));

        // Perform the task
        self.instruct_2(instruction, seed, self.max_tokens(), Some(0.5), stop_at)
    }

    /// Attempt to perform a given task
    pub(crate) fn instruct_2(
        &self,
        instruction: &TokenString,
        seed: u64,
        max_tokens: usize,
        temp: Option<f64>,
        stop_at: &[&str],
    ) -> Result<TokenString> {
        // Create the prompt
        let mut prompt = self.tokenize("### Instruction:\n");
        prompt.push_many(instruction);
        prompt.push_str("\n### Response:\n");

        // Perform inference
        let mut response = self.new_token_string();
        for token in self.infer_iter(prompt, seed, temp, Some(0.9), 1.2, 64)?.take(max_tokens) {
            response.push(token);
            
            // Detokenize the token and break if it matches any stop_at tokens
            if !stop_at.is_empty() {
                let token_str = self.detokenize(&[token]);
                if stop_at.iter().any(|&stop| token_str.ends_with(stop)) {
                    break;
                }
            }
        }

        // Return the response
        Ok(response)
    }

    /// Attempt to get the token for a given string
    pub(crate) fn get_token(&self, s: impl Display) -> Result<u32> {
        match self.tokenizer.get_vocab(true).get(&s.to_string()) {
            Some(token) => Ok(*token),
            None => anyhow::bail!("cannot find the token for {:?}", s.to_string()),
        }
    }

    /// Given a list of items and a context string, try to pick the most appropriate item
    /// based on the context
    pub(crate) fn pick_item(
        &self,
        context: &TokenString,
        desired_traits: Option<&TokenString>,
        items: impl IntoIterator<Item = impl AsRef<str>>,
        seed: u64,
    ) -> Result<String> {
        let items: Vec<String> = items.into_iter().map(|item| item.as_ref().trim().to_lowercase()).collect();

        // Start the prompt with the context
        let mut prompt = self.tokenize("### Context:\n");
        prompt.push_many(context);

        // Add the items to the prompt, in brackets to help guide the model in the next step
        prompt.push_str("\n### Items:\n");
        for item in &items {
            prompt.push_str(&format!("[{}]\n", item));
        }

        // Add the desired trait to the prompt
        if let Some(desired_traits) = desired_traits {
            prompt.push_str("\n### Desired Traits:\n");
            prompt.push_many(desired_traits);
        }
        
        // Ask the model to pick the most appropriate item
        prompt.push_str("\n### Instruction:\nPick the most appropriate item from the list above.\n");

        // Start the response with a bracket to guide the model
        prompt.push_str("\n### Response:\n[");

        // Keep trying until the model picks an item, incrementing the seed each time
        // After the first iteration, temperature is set to 1.0 to encourage diversity
        let mut response = None;
        let mut temperature = 0.0;
        for seed in seed.. {
            // Clone the items
            let mut possible_items = items.clone();
            
            // Begin inference
            let mut inference = self.infer_iter(prompt.clone(), seed, Some(temperature), None, 0.0, 0)?;
            
            // Infer while possible_items > 1
            let mut inferred = String::new();
            while possible_items.len() > 1 {
                // Attempt to get the next token and check if it matches any of the possible items
                if let Some(next_token) = inference.next_token() {
                    // Add the token to the inferred string
                    inferred.push_str(&self.detokenize(&[next_token]));
                    
                    // Remove the item from the list if it doesn't begin with the inferred string
                    let formatted = inferred.trim().to_lowercase();
                    possible_items.retain(|item| item.starts_with(&formatted));
                }
                // If there are no more tokens, empty the possible items and break
                else {
                    possible_items.clear();
                    break;
                }
            }

            // If there is only one item left, return it
            if possible_items.len() == 1 {
                response = Some(possible_items.pop().unwrap());
                break;
            }

            // If the temperature is 0.0, set it to 0.5 and try again
            if temperature == 0.0 {
                temperature = 1.0;
            }
        }
        
        // Return the response
        response.ok_or_else(|| anyhow!("no response"))
    }
}

pub struct InferIter {
    device: Device,
    tokens: TokenString,
    step: usize,
    pipeline: MixFormer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
    reached_eos: bool,
}

impl InferIter {
    pub fn new(
        device: Device,
        tokens: TokenString,
        pipeline: MixFormer,
        logits_processor: LogitsProcessor,
        repeat_penalty: f32,
        repeat_last_n: usize,
        eos_token: u32,
    ) -> Self {
        Self {
            device,
            tokens,
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

        // Get the context
        let context = self.tokens
            .get(self.tokens.len().saturating_sub(context_size)..)
            .unwrap();

        // Create the input tensor containing the context
        let input = Tensor::new(context, &self.device).unwrap().unsqueeze(0).unwrap();

        // Forward the input through the pipeline
        let logits = self.pipeline.forward(&input).unwrap();

        // Get the logits
        let logits = logits.squeeze(0).unwrap().to_dtype(DType::F32).unwrap();

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
            ).unwrap()
        };

        // Sample the next token
        let next_token = self.logits_processor.sample(&logits).unwrap();

        // Increment the step
        self.step += 1;

        // If the token is not the end of text token, add it to the tokens
        if next_token != self.eos_token {
            self.tokens.push(next_token);
        }
        // Otherwise, set reached_eos to true and return None
        else {
            self.reached_eos = true;
            return None;
        }

        // Return the next token
        Some(next_token)
    }
}

impl Iterator for InferIter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

pub enum InferValue {
    String(String),
    Float(f64),
    Int(i64),
}

impl InferValue {
    pub fn to_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Float(f) => f.to_string(),
            Self::Int(i) => i.to_string(),
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        if let Ok(f) = s.parse::<f64>() {
            Ok(Self::Float(f))
        } else if let Ok(i) = s.parse::<i64>() {
            Ok(Self::Int(i))
        } else {
            Ok(Self::String(s.to_string().trim_matches('"').to_string()))
        }
    }
}

impl FromStr for InferValue {
    type Err = E;

    fn from_str(s: &str) -> Result<Self> {
        InferValue::from_str(s)
    }
}

impl Display for InferValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
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

impl<T: Into<InferValue>> From<&T> for InferValue {
    fn from(t: &T) -> Self {
        t.into()
    }
}

#[macro_export]
macro_rules! inference_key_value_pairs {
    [$($key:expr => $value:expr,)*$(,)?] => {
        vec![$(
            ($key, $crate::model::InferValue::from($value)),
        )*]
    };
}