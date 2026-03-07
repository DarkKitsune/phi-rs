use std::fmt::Display;

/// Represents a chat between user and model.
pub struct Chat {
    /// The system prompt.
    system_prompt: String,
    /// The messages in the chat.
    messages: Vec<ChatMessage>,
    /// The model's response is treated as if this text was prepended.
    response_prefix: Option<String>,
}

impl Chat {
    /// Creates a new chat with no messages.
    pub fn new(system_prompt: Option<String>) -> Self {
        Self::from_messages(system_prompt, Vec::new())
    }

    /// Creates a new chat from a list of chat messages.
    pub fn from_messages(system_prompt: Option<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            system_prompt: system_prompt
                .unwrap_or_else(|| "You are a helpful assistant.".to_string()),
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
    pub fn messages(&self) -> &Vec<ChatMessage> {
        &self.messages
    }

    /// Returns the system prompt for the chat.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Set the system prompt for the chat.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.system_prompt = prompt.into();
    }

    /// Returns the response prefix for the chat, if set.
    pub fn response_prefix(&self) -> Option<&str> {
        self.response_prefix.as_deref()
    }

    /// Sets the response prefix for the chat.
    /// The model's response is treated as if this text was prepended before.
    /// This is useful to limit responses to a certain format, size, or content,
    /// especially when used with an equivalent end sequence.
    pub fn set_response_prefix(&mut self, prefix: impl Into<String>) {
        self.response_prefix = Some(prefix.into());
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
pub enum ChatRole {
    User,
    Model,
}
