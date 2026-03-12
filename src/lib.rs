pub mod chat;
pub mod model;
pub mod prelude;
pub mod token_string;

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn chat() {
        const SEED: u64 = 546457;
        const TEMP: f64 = 0.6;
        const CONVERSATION_TURNS: usize = 3;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Start a chat
        let mut chat = Chat::new();
        chat.set_system_prompt("You are a talkative and entertaining person to talk to.");

        // Infer a conversation
        for _ in 0 .. CONVERSATION_TURNS {
            // Generate a user message
            chat.infer_message(ChatRole::User, &model, SEED, Some(TEMP), 1.1, 64, false);
            println!("\n{}", chat.last_message().unwrap());

            // Generate a model message
            chat.infer_message(ChatRole::Model, &model, SEED, Some(TEMP), 1.1, 64, false);
            println!("\n{}", chat.last_message().unwrap());
        }
    }

    #[test]
    fn choose_items() {
        const SEED: u64 = 32623;
        const ATTEMPTS: usize = 7;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Present choices to the model
        let item = model.try_choose_item(
            "a weapon for a knight",
            ["horse", "sword", "potion", "compass", "bow", "shield"],
            SEED,
            0.2,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);

        let item = model.try_choose_item(
            "a soft toy",
            ["snacks", "ball", "coloring book", "stuffed animal", "book"],
            SEED,
            0.2,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);

        let item = model.try_choose_item(
            "the base of spaghetti sauce",
            [
                "bell pepper",
                "tomato",
                "cayenne pepper",
                "ghost pepper",
                "potato",
                "onion",
                "garlic",
                "pineapple",
            ],
            SEED,
            0.2,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);
    }

    #[test]
    fn generate_dog_sentences() {
        const SEED: u64 = 246810;
        const TEMP: f64 = 0.6;
        const NUM_TO_GENERATE: usize = 7;
        const DOG_EXAMPLES: &[&str] = &[
            "The quick brown fox jumps over the lazy dog.",
            "An agile, brown fox vaults over a lethargic canine creature.",
            "A swift, brown fox hops over a sleepy dog.",
            "One lazy dog was jumped over by a quick brown fox.",
            "That fox just jumped over that dang dog!",
            "From what I hear, the red fox that jumped over the dog was very quick.",
            "The red fox quickly jumped over the sleeping dog, startling it awake.",
        ];

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Start dog sentences
        println!(
            "Generating {}  fox jumping over a dog sentences:",
            NUM_TO_GENERATE
        );

        // Iterate and increment the seed to generate multiple similar sentences
        for seed_add in 0..NUM_TO_GENERATE as u64 {
            let generated = model.generate_similar(
                "A sentence describing a fox jumping over a dog",
                DOG_EXAMPLES,
                SEED.wrapping_add(seed_add),
                Some(TEMP),
            );
            println!("{}", generated);
        }
    }

    #[test]
    fn expand_and_summarize() {
        const SEED: u64 = 13579;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        let text = "The quick brown fox jumps over the lazy dog. The cow jumps over the moon.";

        // Expand the text
        let expanded = model.expand(text, SEED, Some(0.6));
        println!("Expanded text: {}", expanded);

        // Summarize the text
        let summary = model.summarize_to_tokens(&expanded, 50, false, SEED, 0.1, 5);
        println!("Summary: {}", summary);
    }
}