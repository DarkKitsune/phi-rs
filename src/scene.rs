use std::fmt::Display;

use crate::{model::Model, token_string::TokenString};

#[derive(Clone)]
pub struct Scene {
    long_term_memory: TokenString,
    short_term_memory: TokenString,
    model: Model,
    characters: Vec<String>,
    last_speaker: Option<String>,
}

impl Scene {
    pub(crate) fn new(
        model: Model,
        setting: impl Display,
        starting_characters: &[impl Display],
    ) -> Self {
        // Concatenate the starting characters into a string, with an oxford comma
        let characters_string =
            starting_characters
                .iter()
                .enumerate()
                .fold(String::new(), |acc, (i, character)| {
                    if i == 0 {
                        character.to_string()
                    } else if i == starting_characters.len() - 1 {
                        format!("{} and {}", acc, character)
                    } else {
                        format!("{}, {}", acc, character)
                    }
                });
        // Put the characters in long term memory
        let long_term_memory = model.tokenize(format!(
            "[{}]\n[There are {} characters: {}]\n",
            setting,
            starting_characters.len(),
            characters_string
        ));
        // The short term memory is empty
        let short_term_memory = model.new_token_string();
        // Create the characters vector
        let characters = starting_characters.iter().map(|c| c.to_string()).collect();
        // Return the new scene
        Self {
            long_term_memory,
            short_term_memory,
            model,
            characters,
            last_speaker: None,
        }
    }

    pub fn long_term_memory(&self) -> &TokenString {
        &self.long_term_memory
    }

    pub fn short_term_memory(&self) -> &TokenString {
        &self.short_term_memory
    }

    pub fn memory_length(&self) -> usize {
        self.long_term_memory.len() + self.short_term_memory.len()
    }

    pub fn get_full_memory(&self) -> TokenString {
        // Combine the long and short term memory
        let mut full_memory = self.long_term_memory.clone();
        full_memory.push_many(&self.short_term_memory);
        // Return the combined memory
        full_memory
    }

    pub fn tokenize(&self, text: impl Display) -> TokenString {
        self.model.tokenize(text)
    }

    pub fn compress_memory(&mut self, if_longer_than: usize) {
        // Exit early if memory_length is less than model.max_tokens() / 2
        if self.memory_length() < if_longer_than {
            return;
        }
        // Get the combined long and short term memory as a token string
        let full_memory = self.get_full_memory();
        // Shorten the full memory and set it as the long term memory
        self.long_term_memory = full_memory.shortened();
        // Clear the short term memory
        self.short_term_memory = self.model.new_token_string();
    }

    pub fn push(&mut self, tokens: &TokenString) {
        // Add the tokens to the short term memory
        self.short_term_memory.push_many(tokens);
        // Add a newline to the short term memory
        self.short_term_memory.push_str("\n");
    }

    pub fn push_story(&mut self, story: impl Display) -> SceneTurn {
        // Format a new line for the story
        let line = format!("{}\n", story);
        // Add the line to the short term memory
        self.short_term_memory.push_str(&line);
        // Return a new scene turn
        SceneTurn::story(story)
    }

    pub fn push_dialogue(&mut self, character: impl Display, dialogue: impl Display) -> SceneTurn {
        // Format a new line for the dialogue
        let line = format!("{}: \"{}\"\n", character, dialogue);
        // Add the line to the short term memory
        self.short_term_memory.push_str(&line);
        // Set the last speaker
        self.last_speaker = Some(character.to_string());
        // Return a new scene turn
        SceneTurn::dialogue(character, dialogue)
    }

    /// Infer a story line and add it to the memory.
    /// Returns the inferred story turn.
    pub fn infer_story(&mut self, max_tokens: usize) -> SceneTurn {
        // Compress the memory if it's getting too long
        self.compress_memory(self.model.max_tokens() / 2);
        // Start the story line with the full memory
        let mut line = self.get_full_memory();
        // Add the beginning of a story line to the full memory
        line.push_str("[");
        // Infer a line from the full memory
        self.push_story(
            line.next(
                max_tokens,
                Some(0.5),
                &[
                    "]", ".]", "?]", "']", ":]", "!]", "\"]", "]\"", "]]", "][", ".\"", "?\"",
                    "!\"", ".", "?", "!",
                ],
            )
            .to_string()
            .replace("]", "")
            .trim(),
        )
    }

    /// Infer a dialogue line and add it to the memory.
    /// Returns the inferred dialogue turn.
    pub fn infer_dialogue(&mut self, character: impl Display, max_tokens: usize) -> SceneTurn {
        // Compress the memory if it's getting too long
        self.compress_memory(self.model.max_tokens() / 2);
        // Start the story line with the full memory
        let mut line = self.get_full_memory();
        // Add the beginning of a dialog line to the full memory
        line.push_str(&format!("{}: \"", character));
        // Infer a line from the full memory
        self.push_dialogue(
            character,
            line.next(max_tokens, Some(0.5), &["\"", ".\"", "?\"", "!\""])
                .to_string()
                .replace("\"", "")
                .trim(),
        )
    }

    /// Infer a random type of turn and add it to the memory.
    /// Automatically decides whether to infer a story or dialogue turn.
    pub fn infer_any(&mut self, max_tokens: usize) -> SceneTurn {
        // Generate a seed from the last 4 tokens of short term memory
        let seed = self
            .short_term_memory()
            .iter()
            .rev()
            .take(4)
            .fold(0u64, |acc, &token| acc.wrapping_add(token as u64))
            .wrapping_add(self.model.seed());
        // Choose the type of turn to infer based on the seed
        // Dialogue turns are 1.5x as likely as story turns
        if seed % 5 < 3 {
            // Choose a character to speak. If the character matches self.previous_turn.speaker(), choose another character.
            for attempt in 0..self.characters.len() {
                let character = self.characters
                    [((!seed) as usize).wrapping_add(attempt) % self.characters.len()]
                .clone();
                if let Some(last_speaker) = &self.last_speaker {
                    if last_speaker != &character {
                        return self.infer_dialogue(character, max_tokens);
                    }
                } else {
                    return self.infer_dialogue(character, max_tokens);
                }
            }
            panic!("No characters to choose from");
        } else {
            self.infer_story(max_tokens)
        }
    }
}

impl Display for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_full_memory())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneTurn {
    turn_type: SceneTurnType,
}

impl SceneTurn {
    pub fn new(turn_type: SceneTurnType) -> Self {
        Self { turn_type }
    }

    pub fn story(story: impl Display) -> Self {
        Self::new(SceneTurnType::Story(story.to_string()))
    }

    pub fn dialogue(character: impl Display, dialogue: impl Display) -> Self {
        Self::new(SceneTurnType::Dialogue(
            character.to_string(),
            dialogue.to_string(),
        ))
    }

    pub fn turn_type(&self) -> &SceneTurnType {
        &self.turn_type
    }
}

impl Display for SceneTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.turn_type {
            SceneTurnType::Story(story) => write!(f, "{}", story),
            SceneTurnType::Dialogue(character, dialogue) => {
                write!(f, "{}: \"{}\"", character, dialogue)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SceneTurnType {
    Story(String),
    Dialogue(String, String),
}
