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
        println!("Generating {}  fox jumping over a dog sentences:", NUM_TO_GENERATE);

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
    fn generate_boss_names() {
        const SEED: u64 = 567869;
        const TEMP: f64 = 0.6;
        const NUM_TO_GENERATE: usize = 7;
        const GAME_BOSS_NAME_EXAMPLES: &[&str] = &[
            "Gorath the Destroyer",
            "Zaldrath the Conqueror",
            "Fallen Angel Idriel",
            "The Dark Sorcerer Malakar"
        ];

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();
        
        // Start game boss names
        println!("Generating {} game boss names:", NUM_TO_GENERATE);
        
        // Iterate and increment the seed to generate multiple game boss names
        for seed_add in 0..NUM_TO_GENERATE as u64 {
            let generated = model.generate_similar(
                "An intimidating but short name for a role-playing game boss",
                GAME_BOSS_NAME_EXAMPLES,
                SEED.wrapping_add(seed_add),
                Some(TEMP),
            );
            println!("{}", generated);
        }
    }
}