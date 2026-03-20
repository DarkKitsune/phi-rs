pub mod chat;
pub mod data;
pub mod inference;
pub mod model;
pub mod model_type;
pub mod prelude;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use ggmath::random::ToSeed;
    use serde_json::json;

    use crate::prelude::*;

    #[test]
    fn chat() {
        const SEED: u64 = 477474;
        const TEMP: f64 = 0.7;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Start a chat
        let mut chat = Chat::new();
        chat.set_system_prompt(
            "You are a helpful assistant and friendly person who has a great imagination, \
        an open mind, and is fun to talk to. Talk to the user in a friendly and engaging manner.",
        );

        println!("System: {}", chat.system_prompt());

        // Infer a conversation
        for turn in 0..CONVERSATION_TURNS {
            // Get the user message from the console input
            let mut input = String::new();
            print!("User: ");
            std::io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim().to_string();
            chat.add_message(ChatRole::User, input);

            // Infer a model response
            chat.infer_message(
                &ChatRole::Model,
                &model,
                false,
                SEED.wrapping_add(turn as u64),
                Some(TEMP),
                1.0,
                0,
                false,
            );
            println!("\n{}\n", chat.last_message().unwrap());
        }
    }

    #[test]
    fn choose_items() {
        const SEED: u64 = 32623;
        const ATTEMPTS: usize = 7;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Present choices to the model
        let item = model.try_choose_item(
            "a weapon for a knight",
            ["horse", "sword", "potion", "compass", "bow", "shield"],
            SEED,
            0.0,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);

        let item = model.try_choose_item(
            "a soft toy",
            ["snacks", "ball", "coloring book", "stuffed animal", "book"],
            SEED,
            0.0,
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
            0.0,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);
    }

    #[test]
    fn generate_dog_sentences() {
        const SEED: u64 = 246810;
        const TEMP: f64 = 0.7;
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
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Start dog sentences
        println!(
            "Generating {} fox jumping over a dog sentences:",
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
        const TEMP: f64 = 0.0;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        let text = "Sally told me she saw that dang fox jump over the poor lazy dog again. A travesty, really.";

        // Expand the text
        let expanded = model.expand(text, SEED, Some(TEMP));
        println!("\nExpanded text: {}", expanded);

        // Summarize the text
        let summary = model.summarize_to_tokens(&expanded, 60, false, SEED, TEMP, 5);
        println!("\nSummary: {}", summary);
    }

    #[test]
    fn predict_next() {
        const SEED: u64 = 13579;
        const TEMP: f64 = 0.7;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        let mut text =
            "The quick brown fox jumps over the lazy dog. The slow brown cow, however, is"
                .to_string();

        // Predict the rest of the text
        text.push_str(
            &model
                .predict_next(&text, SEED, Some(TEMP), None, 1.0, 0)
                .complete(&["\n"])
                .0,
        );
        println!("Completed text: {}", text);
    }

    #[test]
    fn predict_next_chat() {
        const SEED: u64 = 13579;
        const TEMP: f64 = 0.7;
        const CHARACTER_NAMES: &[&str] = &["Alice", "Bob", "Charlie", "Diana"];
        const CONVERSATION_TURNS: usize = 7;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Initialize the string that serves as the chat history
        let mut chat = format!(
            "The scene opens on {} standing in the middle of a forest. The group have encountered a mysterious creature.",
            CHARACTER_NAMES.join(", ")
        );

        // Simulate a conversation
        for turn in 0..(CONVERSATION_TURNS as u64) {
            // Decide which character is speaking this turn
            let speaking_character = CHARACTER_NAMES
                [(SEED.wrapping_add(turn).into_random::<u64>() as usize) % CHARACTER_NAMES.len()];

            // Start the character's dialogue
            chat.push_str(&format!("\n{}: \"", speaking_character));

            // Generate the character's dialogue by predicting the next part of the conversation
            let response = model
                .predict_next(&chat, SEED.wrapping_add(turn), Some(TEMP), None, 1.0, 0)
                .complete(&["\"", "\n"])
                .0;
            chat.push_str(&format!("{}\"", response));
        }

        println!("Final chat:\n{}", chat);
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 6575;
        const TEMP: f64 = 0.7;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Generate a story
        let mut story = "There was once".to_string();
        story.push_str(
            &model
                .predict_next("There was once", SEED, Some(TEMP), None, 1.1, 64)
                .complete(&["\n"])
                .0,
        );
        println!("Generated story:\n{}", story);
    }

    #[test]
    fn thinking_simple_math() {
        const SEED: u64 = 3463;
        const TEMP: f64 = 0.7;

        // Create the model and chat
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Give the model a simple problem to think about
        let result = model
            .instruct("If a train leaves Station A at 60 mph and another leaves Station B 100 miles away at 40 mph towards each other, when do they meet?", true, SEED, Some(TEMP), None, 1.0, 0)
            .complete(&[])
            .0
            .trim()
            .to_string();

        // Parse everything before and after the </think> closing tag
        let parts: Vec<&str> = result.split("</think>").collect();
        let thoughts = parts.get(0).unwrap_or(&"").trim_end();
        let rest = parts.get(1).unwrap_or(&"").trim_start();

        println!("\nThoughts:\n{}\nRest:\n{}\n", thoughts, rest);
    }

    #[test]
    fn thinking_literature() {
        const SEED: u64 = 3463;
        const TEMP: f64 = 0.7;

        // Create the model and chat
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Give the model a literary analysis problem to think about
        let result = model
            .instruct(
                "Analyze the following poem and explain its themes:\n\
                Two roads diverged in a yellow wood,\n\
                And sorry I could not travel both\n\
                And be one traveler, long I stood\n\
                And looked down one as far as I could\n\
                To where it bent in the undergrowth;\n\
                \n\
                Then took the other, as just as fair,\n\
                And having perhaps the better claim,\n\
                Because it was grassy and wanted wear;\n\
                Though as for that the passing there\n\
                Had worn them really about the same,\n\
                \n\
                And both that morning equally lay\n\
                In leaves no step had trodden black.\n\
                Oh, I kept the first for another day!\n\
                Yet knowing how way leads on to way,\n\
                I doubted if I should ever come back.\n\
                \n\
                I shall be telling this with a sigh\n\
                Somewhere ages and ages hence:\n\
                Two roads diverged in a wood, and I—\n\
                I took the one less traveled by,\n\
                And that has made all the difference.'",
                true,
                SEED,
                Some(TEMP),
                None,
                1.0,
                0,
            )
            .complete(&[])
            .0
            .trim()
            .to_string();

        // Parse everything before and after the </think> closing tag
        let parts: Vec<&str> = result.split("</think>").collect();
        let thoughts = parts.get(0).unwrap_or(&"").trim_end();
        let rest = parts.get(1).unwrap_or(&"").trim_start();

        println!("\nThoughts:\n{}\nRest:\n{}\n", thoughts, rest);
    }

    #[test]
    fn ask_json() {
        const SEED: u64 = 3463;
        const TEMP: f64 = 0.0;

        // Create the model
        let model = Model::new(ModelType::Qwen3, SEED, true).unwrap();

        // Create a JSON object
        let json = json!({
            "name": "Alice",
            "pets": [
                {"type": "cat", "name": "Whiskers", "age": 3},
                {"type": "dog", "name": "Fido", "age": 8},
                {"type": "dog", "name": "Rex", "age": 5},
            ],
        });

        // Ask the model a question about the JSON object
        let result = model
            .ask_json(
                json.clone(),
                "What is the name and age of Alice's cat?",
            )
            .complete(&[])
            .0;
        println!("Result:\n{}\n", result);

        // Ask the model to add a new pet to Alice's list of pets
        let json = model
            .edit_json(
                json,
                "Add a new 3-year old parakeet named \"Crackers\" to the list of pets",
                SEED,
                TEMP,
                3,
            )
            .expect("Failed to parse JSON");

        // Ask the model to double the age of all dogs
        let json = model
            .edit_json(
                json,
                "Please double the age of every cat in \"pets\"",
                SEED,
                TEMP,
                3,
            )
            .expect("Failed to parse JSON");

        println!(
            "Edited JSON:\n{}\n",
            serde_json::to_string_pretty(&json).unwrap()
        );
    }
}
