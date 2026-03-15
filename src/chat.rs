use std::{collections::HashMap, fmt::Display};

use crate::{inference::InferValue, model::Model};

/// Represents a chat between user and model.
#[derive(Clone, Debug)]
pub struct Chat {
    /// The system prompt.
    system_prompt: Option<String>,
    /// Long term memory for the chat.
    /// When the chat gets too long, older messages may be summarized and
    /// stored in long term memory. Injected into the system prompt.
    long_term_memory: Option<String>,
    /// Extra data associated with the chat.
    extra_data: Option<HashMap<String, InferValue>>,
    /// The messages in the chat.
    messages: Vec<ChatMessage>,
    /// The model's response is treated as if this text was prepended.
    response_prefix: Option<String>,
}

impl Chat {
    // Estimated number of tokens per character
    const TOKENS_PER_CHARACTER: f64 = 0.25;
    /// Maximum number of tokens for the entire chat, including messages,
    /// system prompt, long term memory and extra data.
    const MAX_TOTAL_TOKENS: usize = 2048;
    /// Maximum number of tokens for the long term memory.
    const MAX_LONG_TERM_MEMORY: usize = 768;
    /// Token count threshold for compressing the chat.
    const COMPRESS_THRESHOLD: usize = const {
        Self::MAX_TOTAL_TOKENS
            .checked_sub(Self::MAX_LONG_TERM_MEMORY)
            .expect("MAX_TOTAL_TOKENS must be greater than MAX_LONG_TERM_MEMORY")
    };
    /// The amount of messages to retain when compressing.
    const COMPRESS_RETAIN_MESSAGES: usize = 2;

    /// Creates a new chat with no messages.
    pub fn new() -> Self {
        Self::from_messages(Vec::new())
    }

    /// Creates a new chat from a list of chat messages.
    pub fn from_messages(messages: Vec<ChatMessage>) -> Self {
        Self {
            system_prompt: None,
            long_term_memory: None,
            extra_data: None,
            messages,
            response_prefix: None,
        }
    }

    /// Push an existing chat message to the chat.
    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    /// Create a new chat message and push it to the chat.
    pub fn add_message(&mut self, sender: ChatRole, content: impl Display) {
        let message = ChatMessage::new(sender, content.to_string());
        self.push(message);
    }

    /// Infer a new chat message using the model and push it to the chat.
    pub fn infer_message(
        &mut self,
        sender: ChatRole,
        model: &Model,
        seed: u64,
        temp: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        ignore_end_sequences: bool,
    ) {
        // If token count exceeds the compress threshold, compress the chat first
        if self.estimate_total_tokens() > Self::COMPRESS_THRESHOLD {
            self.compress(model);
        }

        let end_sequences: &[&str] = if ignore_end_sequences { &[] } else { &["\n\n"] };
        let response = model
            .chat(
                self,
                sender.clone(),
                seed,
                temp,
                None,
                repeat_penalty,
                repeat_last_n,
            )
            .complete(end_sequences)
            .0;

        self.add_message(sender, response);
    }

    /// Returns a reference to the messages in the chat.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    // Returns the last message in the chat.
    pub fn last_message(&self) -> Option<&ChatMessage> {
        self.messages.last()
    }

    /// Returns the system prompt for the chat.
    pub fn system_prompt(&self) -> &str {
        self.system_prompt.as_deref().unwrap_or(
            "You are a helpful assistant whose job is to use what you know to help the user with any problem they have.",
        )
    }

    /// Set the system prompt for the chat.
    pub fn set_system_prompt(&mut self, prompt: impl Display) {
        self.system_prompt = Some(prompt.to_string());
    }

    /// Returns a reference to the long term memory associated with the chat.
    pub fn long_term_memory(&self) -> Option<&str> {
        self.long_term_memory.as_deref()
    }

    /// Returns a reference to the extra data associated with the chat.
    pub fn extra_data(&self) -> Option<&HashMap<String, InferValue>> {
        self.extra_data.as_ref()
    }

    /// Returns a mutable reference to the extra data associated with the chat.
    pub fn extra_data_mut(&mut self) -> &mut HashMap<String, InferValue> {
        if self.extra_data.is_none() {
            self.extra_data = Some(HashMap::new());
        }
        self.extra_data.as_mut().unwrap()
    }

    /// Replaces the extra data associated with the chat using the given map.
    pub fn set_extra_data(&mut self, data: HashMap<String, InferValue>) {
        self.extra_data = Some(data);
    }

    /// Returns the response prefix for the chat, if set.
    pub fn response_prefix(&self) -> Option<&str> {
        self.response_prefix.as_deref()
    }

    /// Sets the response prefix for the chat.
    /// The model's response is treated as if this text was prepended before.
    /// This is useful to limit responses to a certain format, size, or content,
    /// especially when used with an equivalent end sequence.
    pub fn set_response_prefix(&mut self, prefix: Option<String>) {
        self.response_prefix = prefix;
    }

    /// Estimate the total tokens for the entire chat.
    pub fn estimate_total_tokens(&self) -> usize {
        let mut total_tokens = 0;

        if let Some(system_prompt) = &self.system_prompt {
            total_tokens += (system_prompt.len() as f64 * Self::TOKENS_PER_CHARACTER) as usize;
        }
        if let Some(long_term_memory) = &self.long_term_memory {
            total_tokens += (long_term_memory.len() as f64 * Self::TOKENS_PER_CHARACTER) as usize;
        }
        if let Some(extra_data) = &self.extra_data {
            for (key, value) in extra_data {
                total_tokens += (key.len() as f64 * Self::TOKENS_PER_CHARACTER) as usize;
                total_tokens +=
                    (format!("{:?}", value).len() as f64 * Self::TOKENS_PER_CHARACTER) as usize;
            }
        }
        for message in &self.messages {
            total_tokens += (message.content().len() as f64 * Self::TOKENS_PER_CHARACTER) as usize;
        }

        total_tokens
    }

    /// Compresses the chat by removing old messages and summarizing them with the long term memory.
    /// Then, replace the long term memory with this summary.
    pub fn compress(&mut self, model: &Model) {
        // Gather long term memory and all messages up to the last COMPRESS_RETAIN_MESSAGES as chat history
        let mut history = self.long_term_memory().unwrap_or("").to_string();
        for message in &self.messages[..self
            .messages
            .len()
            .saturating_sub(Self::COMPRESS_RETAIN_MESSAGES)]
        {
            history.push_str("\n");
            history.push_str(
                format!(
                    "{}: \"{}\"",
                    model.model_type().chat_role_name(message.sender().clone()),
                    message.content()
                )
                .as_str(),
            );
        }

        println!("Beginning chat compression...");
        println!(
            "Chat history (-{} messages) before compression: {}",
            Self::COMPRESS_RETAIN_MESSAGES,
            history
        );

        // Summarize the chat history using the model
        let summary =
            model.summarize_to_tokens(&history, Self::MAX_LONG_TERM_MEMORY, true, 0, 0.0, 2);

        // Set the long term memory to the summarized chat history
        self.long_term_memory = Some(summary);

        // Clear the messages up to the last COMPRESS_RETAIN_MESSAGES
        let retain_start = self
            .messages
            .len()
            .saturating_sub(Self::COMPRESS_RETAIN_MESSAGES);
        self.messages.drain(..retain_start);

        println!(
            "Chat history after compression:\n{}\n{}",
            self.long_term_memory()
                .unwrap()
                .to_string(),
            self.messages()
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

impl IntoIterator for Chat {
    type Item = ChatMessage;
    type IntoIter = std::vec::IntoIter<ChatMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.into_iter()
    }
}

impl<'a> IntoIterator for &'a Chat {
    type Item = &'a ChatMessage;
    type IntoIter = std::slice::Iter<'a, ChatMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter()
    }
}

/// Represents a single message in a chat.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    sender: ChatRole,
    content: String,
}

impl ChatMessage {
    /// Creates a new chat message.
    pub fn new(sender: ChatRole, content: String) -> Self {
        Self { sender, content }
    }

    /// Returns the sender of the message.
    pub fn sender(&self) -> &ChatRole {
        &self.sender
    }

    /// Returns the content of the message.
    pub fn content(&self) -> &str {
        &self.content
    }
}

impl Display for ChatMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.sender, self.content)
    }
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::User => write!(f, "User"),
            ChatRole::Model => write!(f, "Model"),
        }
    }
}

/// Represents the role of a participant in a chat.
#[derive(Clone, Debug)]
pub enum ChatRole {
    User,
    Model,
}
