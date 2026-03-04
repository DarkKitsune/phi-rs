pub mod crafter;
pub mod chat;
pub mod model;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::model::{Model, ModelType};

    use super::*;
    use crafter::{Crafter, CrafterExample};

    #[test]
    fn crafting() {
        const SEED: u64 = 122534;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Create a crafter
        let crafter = Crafter::new(
            model,
            None,
            &[
                CrafterExample::new(&["water", "fire"], "steam"),
                CrafterExample::new(&["sugar", "water", "bee"], "honey"),
                CrafterExample::new(&["human", "hammer"], "construction worker"),
                CrafterExample::new(&["earth", "water"], "mud"),
                CrafterExample::new(&["clown", "tent"], "circus"),
                CrafterExample::new(&["hope", "despair"], "life"),
            ],
        );

        // Craft some items
        let result = crafter.craft(&["water", "cloud"], SEED);
        println!("water + cloud = {}", result);
        let result = crafter.craft(&["politics", "sword", "bomb"], SEED);
        println!("politics + sword + bomb = {}", result);
        let result = crafter.craft(&["fire", "water"], SEED);
        println!("fire + water = {}", result);
        let result = crafter.craft(&["fire", "water", "earth"], SEED);
        println!("fire + water + earth = {}", result);
    }

    #[test]
    fn choose_items() {
        const SEED: u64 = 545856;
        const ATTEMPTS: usize = 5;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Present choices to the model
        let item = model.try_choose_item(
            "You are a knight in a fantasy world.",
            "The item should be a weapon capable of defeating a dragon.",
            ["horse", "sword", "potion", "compass"],
            SEED,
            ATTEMPTS,
        );
        println!("Chose item: {:?}", item);

        let item = model.try_choose_item(
            "You are a grade school teacher. There is a new student in your class. You want to make them feel welcome.",
            "Something tasty",
            ["snacks", "ball", "coloring book", "stuffed animal", "book"],
            SEED,
            ATTEMPTS
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
        let mut extra = HashMap::new();
        extra.insert(
            "Integers".to_string(),
            INTEGERS
                .into_iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
        extra.insert("Response".to_string(), "[".to_string()); // Start the response with a [ character

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
        const TEMP: f64 = 0.5;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Expand a sentence
        let sentence = "The quick brown fox jumps over the lazy dog.";
        let expanded = model.instruct(
            format!("Expand the following sentence: {}", sentence),
            None,
            SEED,
            Some(TEMP),
            None,
            1.0,
            0,
        )
        .complete(&[]).0;

        println!("Expanded sentence: {}", expanded);
    }

    #[test]
    fn infer_key_value() {
        const SEED: u64 = 643734;
        const TEMP: f64 = 0.5;

        // Create the model
        let model = Model::new(ModelType::Phi15Instruct, SEED, true).unwrap();

        // Create a map representing a person
        let mut map = inference_key_value_pairs! {
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
}
