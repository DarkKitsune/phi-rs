use std::fmt::Display;

use anyhow::Result;

use crate::{
    chat::{Chat, ChatRole},
    model::Model,
};

/// Represents a single action pattern which an `ActionExtractor` can recognize
#[derive(Clone, Debug)]
pub struct ActionPattern {
    name: String,
    arguments: Vec<(String, ArgType)>,
}

impl ActionPattern {
    pub fn new(name: impl Display, arguments: Vec<(String, ArgType)>) -> Self {
        Self { name: name.to_string(), arguments }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &[(String, ArgType)] {
        &self.arguments
    }
}

/// Extracts actions from text by comparing it against known action patterns.
#[derive(Clone)]
pub struct ActionExtractor {
    model: Model,
    patterns: Vec<ActionPattern>,
}

impl ActionExtractor {
    /// Creates a new `ActionExtractor` with no action patterns.
    pub fn new(model: Model) -> Self {
        Self {
            model,
            patterns: Vec::new(),
        }
    }

    /// Adds a new action pattern to the extractor.
    /// # Errors
    ///
    /// Returns an error if the action pattern name is not a valid identifier
    /// or if an action pattern with the same name already exists.

    pub fn add_action_pattern(&mut self, pattern: ActionPattern) -> Result<()> {
        // Verify that the pattern name can be used as a function name
        if !pattern
            .name()
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        {
            anyhow::bail!("Invalid action pattern name: {}", pattern.name());
        }

        // Verify that the pattern name is unique
        if self.patterns.iter().any(|p| p.name() == pattern.name()) {
            anyhow::bail!(
                "Action pattern with name '{}' already exists",
                pattern.name()
            );
        }

        // Add the pattern to the list of known patterns
        self.patterns.push(pattern);

        Ok(())
    }

    /// Returns a slice of all action patterns currently known to the extractor.
    pub fn action_patterns(&self) -> &[ActionPattern] {
        &self.patterns
    }

    /// Extracts an action (if any) from the given text by comparing it against known action patterns.
    pub fn extract_action(&self, text: impl AsRef<str>, attempts: usize) -> Option<Action> {
        // Start chat
        let mut chat = Chat::new();

        // Inform the model about the currently known action patterns
        chat.set_system_prompt(format!(
            "You are a helpful coding assistant who writes function calls representing \
                the given text.\n\n<functions>\n\n{}\n\n</functions>",
            self.patterns
                .iter()
                .map(|p| format!("function fn_{}({}) {{...}}", p.name().trim(), p.arguments().iter().map(|(name, ty)| format!("{}: {}", name, ty.typescript_type())).collect::<Vec<_>>().join(", "))) 
                .collect::<Vec<_>>()
                .join("\n\n")
        ));

        // Ask the model to generate the appropriate function call for the given text
        chat.add_message(
            ChatRole::User,
            format!(
                "What function call would you use to represent the following text?\n{}",
                text.as_ref().trim()
            ),
        );

        // Start off the model's response to force it to stick to the format
        chat.set_response_prefix(Some("fn_".to_string()));

        // Try getting the response from the model up to `attempts` times,
        // only returning a response if it matches one of the action patterns
        let mut found_action = None;
        for i in 0..attempts {
            // Start getting the response back
            let mut action_name_iter = self
                .model
                .chat(&chat, &ChatRole::Model, false, i as u64, None, None, 1.0, 0)
                .0;

            // Keep track of the possible actions that the model might be referring to
            let mut possible_actions: Vec<_> =
                self.patterns.iter().map(ActionPattern::name).collect();

            // Keep getting the next token from the model's response until there is one possible action left
            let mut action_name_so_far = String::new();
            while possible_actions.len() > 1 {
                if let Some(next_token) = action_name_iter.next() {
                    action_name_so_far.push_str(&self.model.detokenize(&[next_token]));
                    possible_actions
                        .retain(|action_name| action_name.starts_with(action_name_so_far.trim()));
                } else {
                    // If the model's response has ended but there are still multiple possible actions,
                    // then break out of the loop
                    break;
                }
            }

            // If there is exactly one possible action left, return it as the extracted action
            if possible_actions.len() == 1 {
                // Store the name
                let found_action_name = possible_actions[0].to_string();

                // Get the arguments by editing the response prefix and inferring til )
                chat.set_response_prefix(Some(format!("fn_{}(", found_action_name)));
                let argument_string = self
                    .model
                    .chat(&chat, &ChatRole::Model, false, i as u64, None, None, 1.0, 0)
                    .0
                    .complete(&[")", "\n"])
                    .0;

                // Parse the argument string into individual arguments
                let arguments = argument_string
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<_>>();

                found_action = Some(Action::new(found_action_name, arguments));

                break;
            }
        }

        // Return the found action, if any
        found_action
    }
}

/// An action extracted from text, consisting of a name and a list of arguments.
#[derive(Clone, Debug)]
pub struct Action {
    name: String,
    arguments: Vec<String>,
}

impl Action {
    pub fn new(name: String, arguments: Vec<String>) -> Self {
        Self { name, arguments }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.name, self.arguments.join(", "))
    }
}

/// The type of an argument to an action.
#[derive(Clone, Copy, Debug)]
pub enum ArgType {
    String,
    Number,
    Boolean,
}

impl ArgType {
    pub fn typescript_type(&self) -> &'static str {
        match self {
            ArgType::String => "string",
            ArgType::Number => "number",
            ArgType::Boolean => "boolean",
        }
    }

}