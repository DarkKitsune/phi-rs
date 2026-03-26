pub mod action;
pub mod actor;
pub mod chat;
pub mod data;
pub mod inference;
pub mod model;
pub mod model_type;
pub mod prelude;
pub mod scene;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::json;

    use crate::prelude::*;

    #[test]
    fn chat() {
        const SEED: u64 = 477474;
        const TEMP: f64 = 0.6;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

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
                1.1,
                64,
                false,
            );
            println!("\n{}\n", chat.last_message().unwrap());
        }
    }

    #[test]
    fn scene() {
        const SEED: u64 = 12345;
        const TEMP: f64 = 0.6;
        const STORY_CYCLES: usize = 5;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Create a new scene
        let mut scene = Scene::new(
            "Forest Encounter",
            "The scene opens in a dense forest. Alice and Bob are exploring the area, \
            looking for signs of a supposed nearby ruin.",
            model,
        );

        // Add actors to the scene
        scene.add_actor(Actor::new(
            "Alice",
            "A curious and adventurous young woman, with a knack for archery.",
        ));
        scene.add_actor(Actor::new(
            "Bob",
            "A cautious and thoughtful young man, always looking out for his friends.",
        ));

        // Add a turn to the scene
        scene
            .add_turn(SceneTurn::dialogue("Alice", "What is that over there?"))
            .unwrap();
        scene
            .add_turn(SceneTurn::dialogue(
                "Bob",
                "I think it's a mysterious creature.",
            ))
            .unwrap();

        for _ in 0..STORY_CYCLES {
            scene
                .infer_next_turn(InferredSceneTurn::story(), SEED, Some(TEMP), 1.1, 64)
                .unwrap();

            scene
                .infer_next_turn(
                    InferredSceneTurn::action("Alice".to_string()),
                    SEED,
                    Some(TEMP),
                    1.1,
                    64,
                )
                .unwrap();

            scene
                .infer_next_turn(
                    InferredSceneTurn::dialogue("Bob".to_string()),
                    SEED,
                    Some(TEMP),
                    1.1,
                    64,
                )
                .unwrap();

            println!("\n{}\n", scene);
        }

        // Print the scene
        println!("Scene:\n{}", scene);
    }

    #[test]
    fn action_extraction() {
        const SEED: u64 = 3525;
        const ATTEMPTS: usize = 5;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Create an action extractor
        let mut extractor = ActionExtractor::new(model.clone());

        // Add some action patterns
        extractor
            .add_action_pattern(ActionPattern::new(
                "travel",
                vec!["direction".to_string()],
            ))
            .unwrap();
        extractor
            .add_action_pattern(ActionPattern::new(
                "attack",
                vec!["target_name".to_string()],
            ))
            .unwrap();
        extractor
            .add_action_pattern(ActionPattern::new(
                "talk",
                vec!["dialog_string".to_string()],
            ))
            .unwrap();

        // Extract some actions from text
        let text_strings = [
            "Go north",
            "Go south",
            "Attack the goblin",
            "Say hello to Jimmy",
            "Kill the villagers",
            "Walk to the east",
        ];

        for text in text_strings {
            let action = extractor.extract_action(text, ATTEMPTS);
            println!(
                "Extracted action from '{}': {}",
                text,
                action
                    .as_ref()
                    .map(Action::to_string)
                    .unwrap_or("None".to_string())
            );
        }
    }

    #[test]
    fn choose_items() {
        const SEED: u64 = 32623;
        const ATTEMPTS: usize = 7;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Present choices to the model
        let item = model.try_choose_item(
            "a weapon for a knight",
            ["horse", "sword", "potion", "compass", "bow", "shield"],
            SEED,
            None,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);

        let item = model.try_choose_item(
            "a soft toy",
            ["snacks", "ball", "coloring book", "stuffed animal", "book"],
            SEED,
            None,
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
            None,
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
        const SEED: u64 = 75863;
        const TEMP: f64 = 0.0;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        let text = "Sally told me she saw that dang fox jump over the poor lazy dog again. A travesty, really. \
            Maybe someone should do something about it?";

        // Expand the text
        let expanded = model.expand(text, SEED, Some(TEMP));
        println!("\nExpanded text: {}", expanded);

        // Summarize the text
        let summary = model.summarize(&expanded, SEED, Some(TEMP));
        println!("\nSummary: {}", summary);
    }

    #[test]
    fn predict_chain() {
        const SEED: u64 = 13579;
        const TEMP: f64 = 0.6;
        const TEMPLATE: &str = "There once was a man named {} who lived in a {}.";

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        let (generated, full_text) = model.predict_chain(TEMPLATE, SEED, Some(TEMP), None, 1.1, 64);
        for (i, text) in generated.iter().enumerate() {
            println!("Generated text {}: {}", i, text);
        }
        println!("Full text: {}", full_text);
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 547845;
        const TEMP: f64 = 0.6;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Generate a story
        let mut story = "There was once".to_string();
        story.push_str(
            &model
                .predict_next(
                    "long_story = \"There was once",
                    SEED,
                    Some(TEMP),
                    None,
                    1.1,
                    64,
                )
                .complete(&["\""])
                .0,
        );
        println!("Generated story:\n{}", story);
    }

    #[test]
    fn thinking_simple_math() {
        const SEED: u64 = 3463;
        const TEMP: f64 = 0.6;

        // Create the model and chat
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Give the model a simple problem to think about
        let (result, thoughts) = model.instruct(
            "If a train leaves Station A at 60 mph and another leaves Station B 100 \
                miles away at 40 mph towards each other, when do they meet?",
            true,
            SEED,
            Some(TEMP),
            None,
            1.1,
            64,
        );
        let result = result.complete(&[]).0.trim().to_string();

        println!(
            "\nThoughts:\n{}\nRest:\n{}\n",
            thoughts.unwrap_or_default(),
            result
        );
    }

    #[test]
    fn ask_json() {
        const SEED: u64 = 635681;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

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
            .ask_json(json.clone(), "What is the name and age of Alice's cat?")
            .complete(&[])
            .0;

        println!(
            "What is the name and age of Alice's cat?\nResult:\n{}\n",
            result
        );

        // Ask the model to add a new pet to Alice's list of pets
        let json = model
            .edit_json(
                json,
                "Add a new 3-year old parakeet named \"Crackers\" to the list of pets",
                SEED,
                None,
                3,
            )
            .expect("Failed to parse JSON");

        println!(
            "JSON with parakeet added:\n{}\n",
            serde_json::to_string_pretty(&json).unwrap()
        );

        // Ask the model how many birds are in Alice's list of pets
        let result = model
            .ask_json(json.clone(), "How many birds are in Alice's list of pets?")
            .complete(&[])
            .0;

        println!(
            "How many birds are in Alice's list of pets?\nResult:\n{}\n",
            result
        );
    }
}
