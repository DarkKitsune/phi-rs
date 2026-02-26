use std::{collections::HashMap, fmt::Display};

use itertools::Itertools;

use crate::model::Model;

/// Use a Model to infer the results of crafting
/// two or more items together.
pub struct Crafter {
    model: Model,
    temp: Option<f64>,
    examples: String,
}

impl Crafter {
    pub fn new<'a>(
        model: Model,
        temp: Option<f64>,
        examples: impl IntoIterator<Item = &'a CrafterExample>,
    ) -> Self {
        // Create a string with the examples separated by newlines
        let examples = examples
            .into_iter()
            .map(|example| format!("Combining {} results in {}", example.items, example.result))
            .join("\n");

        Self {
            model,
            temp,
            examples,
        }
    }

    pub fn craft(&self, items: impl IntoIterator<Item = impl Display>, seed: u64) -> String {
        // Join the items with a "] + [" separator
        let joined_items = items.into_iter().join("] + [");

        // Generate the instruction
        let instruction = format!(
            "What might you get by combining [{}]? Be creative and use the examples.",
            joined_items
        );

        // Put examples in extra information
        let mut extra: HashMap<String, String> = HashMap::new();
        extra.insert("Known Combinations".into(), self.examples.clone());

        // Start the response off to help the model
        let response_prefix = format!("If you combine [{}] you get: [", joined_items);
        extra.insert("Response".into(), response_prefix);

        // Gather the results until a ] is found
        self.model
            .instruct(
                &instruction,
                Some(&extra),
                seed,
                Some(self.temp.unwrap_or(0.0)),
                None,
                1.0,
                0,
            )
            .complete_until(&["]"])
            .0
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
