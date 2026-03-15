use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Model as Phi};
use candle_transformers::models::qwen2::ModelForCausalLM as Qwen2;
use candle_transformers::models::qwen3::ModelForCausalLM as Qwen3;
use hf_hub::api::sync::ApiRepo;

use crate::chat::{Chat, ChatRole};
use crate::model::{DynConfig, Pipeline};

/// Represents the type of model to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    // PhiHermes,
    Phi15Instruct,
    Qwen25Instruct,
    Qwen3,
}

impl ModelType {
    pub fn can_chat(&self) -> bool {
        match self {
            ModelType::Phi15Instruct
            | ModelType::Qwen25Instruct => true,
            _ => false,
        }
    }

    pub fn can_think(&self) -> bool {
        match self {
            ModelType::Qwen3 | ModelType::Qwen25Instruct => true,
            _ => false,
        }
    }

    pub fn model_repo(&self) -> &'static str {
        match self {
            // ModelType::PhiHermes => "lmz/candle-quantized-phi",
            ModelType::Phi15Instruct => "rasyosef/Phi-1_5-Instruct-v0.1",
            ModelType::Qwen25Instruct => "Qwen/Qwen2.5-1.5B-Instruct",
            ModelType::Qwen3 => "Qwen/Qwen3-1.7B",
        }
    }

    pub fn tokenizer_repo(&self) -> &'static str {
        match self {
            // ModelType::PhiHermes => "lmz/candle-quantized-phi",
            ModelType::Phi15Instruct => "rasyosef/Phi-1_5-Instruct-v0.1",
            ModelType::Qwen25Instruct => "Qwen/Qwen2.5-1.5B-Instruct",
            ModelType::Qwen3 => "Qwen/Qwen3-1.7B",
        }
    }

    pub fn tokenizer_json_name(&self) -> &'static str {
        match self {
            // ModelType::PhiHermes => "tokenizer-puffin-phi-v2.json",
            _ => "tokenizer.json",
        }
    }

    pub fn model_names(&self) -> &[&'static str] {
        match self {
            // ModelType::PhiHermes => "model-phi-hermes-1_3B.safetensors",
            ModelType::Qwen3 => &["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            _ => &["model.safetensors"],
        }
    }

    /// Load and create a config for this type of model.
    pub fn create_config(&self, repo: &ApiRepo) -> DynConfig {
        match self {
            ModelType::Phi15Instruct => {
                let config_filename = repo.get("config.json").unwrap();
                let config = std::fs::read_to_string(config_filename).unwrap();
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Phi(config)
            },
            ModelType::Qwen25Instruct => {
                let config_filename = repo.get("config.json").unwrap();
                let config = std::fs::read_to_string(config_filename).unwrap();
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Qwen2(config)
            },
            ModelType::Qwen3 => {
                let config_filename = repo.get("config.json").unwrap();
                let config = std::fs::read_to_string(config_filename).unwrap();
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Qwen3(config)
            }
        }
    }

    /// Create a pipeline for this type of model.
    pub fn create_pipeline(&self, config: &DynConfig, var: VarBuilder) -> Pipeline {
        match self {
            ModelType::Phi15Instruct => {
                Pipeline::Phi(Phi::new(config.as_phi().unwrap(), var).unwrap())
            },
            ModelType::Qwen25Instruct => {
                Pipeline::Qwen2(Qwen2::new(config.as_qwen2().unwrap(), var).unwrap())
            },
            ModelType::Qwen3 => {
                Pipeline::Qwen3(Qwen3::new(config.as_qwen3().unwrap(), var).unwrap())
            },
        }
    }

    /// Preprocesses the logits for this model type.
    pub fn process_logits(&self, logits: Tensor) -> Tensor {
        match self {
            ModelType::Phi15Instruct => {
                // Process logits for Phi15Instruct model
                logits.squeeze(0).unwrap().to_dtype(DType::F32).unwrap()
            }
            ModelType::Qwen25Instruct | ModelType::Qwen3 => {
                // Process logits for Qwen3 model
                logits.squeeze(0).unwrap().squeeze(0).unwrap().to_dtype(DType::F32).unwrap()
            }
        }
    }


    /// Creates a chat prompt meant for this type of Phi model.
    pub fn create_chat_prompt(&self, chat: &Chat, sender: ChatRole) -> String {
        let mut prompt = match self {
            // ModelType::PhiHermes => Self::create_phi_hermes_chat_prompt(chat),
            _ => Self::create_chatml_instruct_chat_prompt(chat, sender),
        };

        // Append the response prefix
        if let Some(prefix) = chat.response_prefix() {
            prompt.push_str(prefix);
        }

        prompt
    }

    fn create_chatml_instruct_chat_prompt(chat: &Chat, sender: ChatRole) -> String {
        let mut prompt = String::new();

        // Add the system prompt to the system section
        prompt.push_str(&format!("<|im_start|>system\n{}\n", chat.system_prompt(),));

        // Add the long term memory to the system section
        if let Some(long_term_memory) = chat.long_term_memory() {
            prompt.push_str(long_term_memory);
            prompt.push('\n');
        }

        // Add extra data as key-value pairs for the model to understand
        if let Some(extra_data) = chat.extra_data() {
            prompt.push_str(&format!(
                "What you know: {{\n{}\n}}\n",
                extra_data
                    .iter()
                    .map(|(k, v)| format!("\"{}\" = {}", k, v))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }

        // Finally end the system section
        prompt.push_str("<|im_end|>\n");

        // Add each message in the chat to the prompt as a new section
        for message in chat {
            match message.sender() {
                ChatRole::User => {
                    prompt.push_str(&format!(
                        "<|im_start|>user\n{}\n<|im_end|>\n",
                        message.content()
                    ));
                }
                ChatRole::Model => {
                    prompt.push_str(&format!(
                        "<|im_start|>assistant\n{}\n<|im_end|>\n",
                        message.content()
                    ));
                }
            }
        }

        // Start the final assistant section (response)
        prompt.push_str("<|im_start|>");
        prompt.push_str(match sender {
            ChatRole::User => "user",
            ChatRole::Model => "assistant",
        });
        prompt.push_str("\n");

        prompt
    }

    pub fn chat_role_name(&self, role: ChatRole) -> &'static str {
        match self {
            _ => match role {
                ChatRole::Model => "Assistant",
                ChatRole::User => "User",
            },
        }
    }
}