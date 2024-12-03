use std::{collections::HashMap, fmt::Display};

use itertools::Itertools;

use crate::model::{InferIter, Model};

/// Use a Model to infer the results of crafting
/// two or more items together.
pub struct Crafter {
    model: Model,
    temp: Option<f64>,
    examples: String,
}

impl Crafter {
    pub fn new<'a>(model: Model, temp: Option<f64>, examples: impl IntoIterator<Item = &'a CrafterExample>) -> Self {
        // Create a string with the examples separated by newlines
        let examples = examples
            .into_iter()
            .map(|example| {
                format!(
                    "When you combine {} you get {}.",
                    example.items, example.result
                )
            })
            .join("\n");

        Self { model, temp, examples }
    }

    pub fn craft(&self, items: impl IntoIterator<Item = impl Display>, seed: u64) -> InferIter {
        // Generate the instruction
        let instruction = format!("Given the examples, what might you get by combining [{}]?", items.into_iter().join("] + ["));

        // Put examples in extra information
        let mut extra = HashMap::new();
        extra.insert("Examples", self.examples.as_ref());

        // Start the response off with a [ character
        extra.insert("Response", "[");

        // Infer the result
        self.model.instruct(&instruction, Some(&extra), seed, self.temp, Some(0.9), 1.0, 0)
    }
}

pub struct CrafterExample {
    pub items: String,
    pub result: String,
}

impl CrafterExample {
    pub fn new(items: impl IntoIterator<Item = impl Display>, result: impl Display) -> Self {
        // Format the items as a string with a "+" separator
        let items = format!("[{}]", items.into_iter().join("] + ["));

        Self {
            items,
            result: format!("[{}]", result),
        }
    }
}
