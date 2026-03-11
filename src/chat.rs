use std::{collections::HashMap, fmt::Display};

use crate::model::InferValue;

/// Represents a chat between user and model.
#[derive(Clone, Debug)]
pub struct Chat {
    /// The system prompt.
    system_prompt: Option<String>,
    /// Extra data associated with the chat.
    extra_data: Option<HashMap<String, InferValue>>,
    /// The messages in the chat.
    messages: Vec<ChatMessage>,
    /// The model's response is treated as if this text was prepended.
    response_prefix: Option<String>,
}

impl Chat {
    /// Creates a new chat with no messages.
    pub fn new() -> Self {
        Self::from_messages(Vec::new())
    }

    /// Creates a new chat from a list of chat messages.
    pub fn from_messages(messages: Vec<ChatMessage>) -> Self {
        Self {
            system_prompt: None,
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

    /// Returns a reference to the messages in the chat.
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Returns the system prompt for the chat.
    pub fn system_prompt(&self) -> &str {
        self.system_prompt.as_deref().unwrap_or(
            "You are a helpful assistant whose job is to use what you know to
help the user with any problem they have."
        )
    }

    /// Set the system prompt for the chat.
    pub fn set_system_prompt(&mut self, prompt: impl Display) {
        self.system_prompt = Some(prompt.to_string());
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

/// Represents the role of a participant in a chat.
#[derive(Clone, Debug)]
pub enum ChatRole {
    User,
    Model,
}
