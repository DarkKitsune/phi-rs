use std::collections::HashMap;
use std::fmt::Display;
use std::str::FromStr;

use anyhow::{Error as E, Result};

use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::token_string::{IntoTokenString, TokenString};

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

    pub fn tokenize_str(&self, text: impl Display) -> TokenString {
        // Tokenize the text
        let tokens = self.tokenizer.encode(text.to_string(), true).unwrap();

        // Get the token ids
        let token_ids = tokens.get_ids().to_vec();
        TokenString::new(token_ids, self.clone())
    }

    pub fn tokenize(&self, text: impl IntoTokenString) -> TokenString {
        text.into_token_string(self)
    }

    pub(crate) fn detokenize(&self, tokens: impl AsRef<[u32]>) -> String {
        // Decode the tokens into a string
        let text = self.tokenizer.decode(tokens.as_ref(), true).map_err(E::msg).unwrap();
        text
    }

    /// Attempt to get the token for a given string
    pub(crate) fn get_token(&self, s: impl Display) -> Result<u32> {
        match self.tokenizer.get_vocab(true).get(&s.to_string()) {
            Some(token) => Ok(*token),
            None => anyhow::bail!("cannot find the token for {:?}", s.to_string()),
        }
    }

    /// Get an iterator that yields tokens generated by the model.
    /// Returns an error if the prompt is empty.
    pub fn infer_iter(
        &self,
        prompt: impl IntoTokenString,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<InferIter> {
        // Add the model seed to the seed provided
        let seed = seed.wrapping_add(self.seed);

        // Tokenize the prompt
        let prompt = self.tokenize(prompt);

        // Fail if the prompt is empty
        if prompt.is_empty() {
            anyhow::bail!("prompt was empty")
        }

        // Create pipeline
        let pipeline = MixFormer::new(&self.config, self.vb.clone()).unwrap();

        // Create logits processor
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        // Get the end of text token
        let eos_token = self.get_token("<|endoftext|>").unwrap();

        // Create the iterator
        Ok(InferIter::new(
            self.device.clone(),
            prompt,
            pipeline,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            eos_token,
        ))
    }

    /// Convenience function to create a prompt for instruct
    fn create_instruct_prompt(
        &self,
        instruction: impl AsRef<str>,
        extra_information: Option<&HashMap<&str, impl AsRef<str>>>,
    ) -> TokenString {
        // Start the prompt
        let mut prompt = String::new();

        // Add the extra information to the prompt
        if let Some(extra_information) = extra_information {
            // For each key-value pair, add it to the prompt
            for (key, value) in extra_information {
                // Skip the "Response" key
                if *key == "Response" {
                    continue;
                }
                prompt.push_str(&format!("### {}:\n{}\n", key, value.as_ref()));
            }
        }

        // Add the instruction to the prompt
        prompt.push_str(&format!("### Instruction:\n{}\n", instruction.as_ref()));

        // Ask the model to generate the response
        prompt.push_str("### Response:\n");

        // If extra_information has a "Response" key, add it to the prompt
        if let Some(response) = extra_information.and_then(|h| h.get("Response")) {
            prompt.push_str(response.as_ref());
        }

        self.tokenize(prompt)
    }

    /// Instruct the model to generate a response based on the instruction.
    /// Returns an iterator that can be used to get the tokens generated by the model.
    pub fn instruct(
        &self,
        instruction: impl AsRef<str>,
        extra_information: Option<&HashMap<&str, impl AsRef<str>>>,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> InferIter {
        // Create the prompt
        let prompt = self.create_instruct_prompt(instruction, extra_information);

        // Begin inference
        self.infer_iter(prompt, seed, temp, top_p, repeat_penalty, repeat_last_n).unwrap()
    }

    /// Given a list of items and a context string, try to choose the most appropriate item
    /// based on the context.
    /// Returns the chosen item (lowercased and trimmed) if successful, otherwise None.
    pub fn try_choose_item(
        &self,
        context: impl AsRef<str>,
        desired_traits: impl AsRef<str>,
        items: impl IntoIterator<Item = impl AsRef<str>>,
        seed: u64,
        attempts: usize,
    ) -> Option<String> {
        let mut prompt_extra = HashMap::new();

        // Make sure that seed + attempts doesn't overflow by subtracting u64::MAX / 2
        let seed = if seed > u64::MAX - attempts as u64 {
            seed - u64::MAX / 2
        } else {
            seed
        };

        // Trim and lowercase all the items
        let items: Vec<String> = items
            .into_iter()
            .map(|item| item.as_ref().trim().to_lowercase())
            .collect();

        // Format the items like so: "[item1], [item2], [item3]"
        let items_string = format!("[{}]", items.join("]["));

        // Add the context, items string and desired traits to the extra information
        prompt_extra.insert("Context", context.as_ref());
        prompt_extra.insert("Items", items_string.as_ref());
        prompt_extra.insert("Desired Traits", desired_traits.as_ref());

        // Start the response with a [ character
        prompt_extra.insert("Response", "[");
        
        // Create the prompt
        let prompt = self.create_instruct_prompt(
            "Choose the most appropriate item for the context and desired traits.",
            Some(&prompt_extra),
        );

        // Keep trying until the model chooses an item, incrementing the seed each time
        // After each attempt, temperature is increased to encourage diversity
        let mut response = None;
        let mut temperature = 0.2;
        for seed in seed..seed + attempts as u64 {
            // Clone the items
            let mut possible_items = items.clone();
            
            // Begin inference
            let mut inference = self.infer_iter(prompt.clone(), seed, Some(temperature), None, 1.0, 0).unwrap();
            
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

            // Raise the temperature and try again
            temperature += 0.2;
        }
        
        // Return the response
        response
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
    pub(crate) fn new(
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

    /// Run the iterator until completion and return the remaining tokens as a `TokenString`
    pub fn complete(mut self) -> TokenString {
        let mut response = self.tokens.model.new_token_string();
        while let Some(token) = self.next_token() {
            response.push_token(token);
        }

        response
    }

    /// Run the iterator until completion or until `end_string` is generated
    /// and return everything up to that point as a `String`
    pub fn complete_until(mut self, end_string: impl AsRef<str>) -> String {
        let end_string = end_string.as_ref();
        let mut response = String::new();
        while let Some(token) = self.next_token() {
            let token_str = self.tokens.model.detokenize(&[token]);
            if token_str.contains(end_string) {
                response.push_str(token_str.split(end_string).next().unwrap());
                break;
            }
            response.push_str(&token_str);
        }

        response
    }
}

impl Iterator for InferIter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

impl Into<TokenString> for InferIter {
    fn into(self) -> TokenString {
        self.complete()
    }
}

impl Into<String> for InferIter {
    fn into(self) -> String {
        self.complete().to_string()
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