pub mod chat;
pub mod model;
pub mod prelude;
pub mod token_string;

#[cfg(test)]
mod tests {
    use crate::prelude::*;

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
    fn sort_integers() {
        const INTEGERS: [i32; 6] = [34, 7, 23, 32, 5, 62];
        const SEED: u64 = 987654;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Extra data to pass to the model
        let mut extra = data_map!();
        extra.insert(
            "Integers".to_string(),
            INTEGERS
                .into_iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
                .into(),
        );
        extra.insert("Response".to_string(), "[".into()); // Start the response with a [ character

        // Get the sorted integers from the model
        let response = model
            .instruct(
                "Sort the integers in ascending order.",
                Some(&extra),
                SEED,
                Some(0.0),
                None,
                1.0,
                0,
            )
            .complete(&["]"])
            .0;

        println!("Sorted integers: [{}]", response);
    }

    #[test]
    fn expand_sentence() {
        const SEED: u64 = 123456;
        const TEMP: f64 = 0.3;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Expand a sentence
        let sentence = "The quick brown fox jumps over the lazy dog.";
        let expanded = model
            .instruct(
                format!("Expand the following sentence: {}", sentence),
                None,
                SEED,
                Some(TEMP),
                None,
                1.0,
                0,
            )
            .complete(&["\n"])
            .0;

        println!("Expanded sentence: {}", expanded);
    }

    #[test]
    fn summarize_long_text() {
        const SEED: u64 = 42069;
        const TEMP: f64 = 0.3;
        const TEXT: &str = "Once upon a time in a faraway land, there was a small village surrounded by mountains and forests. The villagers lived a simple life, tending to their farms and animals, and sharing stories by the fire at night. One day, a mysterious traveler arrived, bringing news of a hidden treasure buried deep within the mountains. The villagers were both excited and fearful, unsure if they should seek the treasure or leave it undisturbed. They decided to hold a council to discuss the matter. At the council, they debated the pros and cons of searching for the treasure, considering the potential dangers and rewards. They ultimately decided to send a small group of brave villagers to search for the treasure, while the rest of the village continued their daily lives, hoping for the best.";

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Instruct the model to summarize the text
        let summary = model
            .instruct(
                format!("Very briefly summarize the following text: {}", TEXT),
                None,
                SEED,
                Some(TEMP),
                None,
                1.0,
                0,
            )
            .complete(&["\n"])
            .0;

        println!("Summary: {}", summary);
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 987654;
        const TEMP: f64 = 0.7;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Generate a short story
        let prompt = "Write a short story about a brave knight who saves a village from a dragon.";
        let story = model
            .instruct(prompt, None, SEED, Some(TEMP), None, 1.0, 0)
            .complete(&[])
            .0;

        println!("Generated story: {}", story);
    }

    #[test]
    fn algebra_problem() {
        const SEED: u64 = 135792;
        const TEMP: f64 = 0.1;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Solve an algebra problem
        let problem = "2x + 3 = 7";
        let solution = model
            .instruct(
                format!("Solve the following algebra problem for x: {}", problem),
                None,
                SEED,
                Some(TEMP),
                None,
                1.0,
                0,
            )
            .complete(&[])
            .0;

        println!("Solution to algebra problem: {}", solution);
    }

    #[test]
    fn infer_key_value() {
        const SEED: u64 = 643734;
        const TEMP: f64 = 0.5;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Create a map representing a person
        let mut map = data_map! {
            "age" => 21,
            "class" => "Knight",
            "hometown" => "Millwood",
        };

        // Generate the person's name, and add it to the map
        let name: String = model
            .try_infer_key_value(&map, "name", SEED, TEMP, 7)
            .unwrap();
        map.insert("name".to_string(), name.into());

        // Generate the person's weapon, and add it to the map
        let weapon: String = model
            .try_infer_key_value(&map, "weapon", SEED, TEMP, 7)
            .unwrap();
        map.insert("weapon".to_string(), weapon.into());

        // Generate the person's armor, and add it to the map
        let armor: String = model
            .try_infer_key_value(&map, "armor", SEED, TEMP, 7)
            .unwrap();
        map.insert("armor".to_string(), armor.into());

        // Generate the person's weapon damage, and add it to the map
        let weapon_damage: usize = model
            .try_infer_key_value(&map, "weapon_damage", SEED, TEMP, 7)
            .unwrap();
        map.insert("weapon_damage".to_string(), weapon_damage.into());

        // Generate the person's health, and add it to the map
        let health: usize = model
            .try_infer_key_value(&map, "health", SEED, TEMP, 7)
            .unwrap();
        map.insert("health".to_string(), health.into());

        // Print the final map
        println!("Final map: {:#?}", map);
    }

    #[test]
    fn generate_from_examples() {
        const SEED: u64 = 246810;
        const TEMP: f64 = 0.7;
        const NUM_TO_GENERATE: usize = 20;
        const EXAMPLES: &[&str] = &[
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

        println!("Generating {} similar sentences:", NUM_TO_GENERATE);

        // Iterate and increment the seed to generate multiple similar sentences
        for seed_add in 0..NUM_TO_GENERATE as u64 {
            let generated = model.generate_similar(
                "A sentence describing a fox jumping over a dog",
                EXAMPLES,
                SEED.wrapping_add(seed_add),
                Some(TEMP),
                None,
                1.0,
                0,
            );
            println!("{}", generated);
        }
    }
}
