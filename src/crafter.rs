use std::fmt::Display;

use itertools::Itertools;

use crate::model::Model;

/// Use a Model to infer the results of crafting
/// two or more items together.
pub struct Crafter {
    model: Model,
    examples: String,
}

impl Crafter {
    pub fn new<'a>(model: Model, examples: impl IntoIterator<Item = &'a CrafterExample>) -> Self {
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

        Self { model, examples }
    }

    pub fn craft(&self, items: impl IntoIterator<Item = impl Display>) -> String {
        // Create the instruction token string
        let instruction = self.model.tokenize(format!(
            "{}\nWhat item do you get when you combine {}? Use only one or two words, keep it short but creative.",
            self.examples,
            items.into_iter().join(" and "),
        ));

        // Create the seed
        let seed = self.model.seed();

        // Infer the result
        instruction
            .instruct_2(seed, 64, Some(0.5), &[".", ".\""])
            .to_string()
            .trim_matches('"')
            .trim_end_matches('.')
            .trim_matches('"')
            .to_string()
    }
}

pub struct CrafterExample {
    pub items: String,
    pub result: String,
}

impl CrafterExample {
    pub fn new(items: impl IntoIterator<Item = impl Display>, result: impl Display) -> Self {
        // Format the items as a string with a "+" separator
        let items = items.into_iter().join(" and ");

        Self {
            items,
            result: result.to_string(),
        }
    }
}
