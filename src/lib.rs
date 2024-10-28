pub mod crafter;
pub mod model;
pub mod scene;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::crafter::{Crafter, CrafterExample};

    use super::*;
    use model::Model;

    #[test]
    fn scene() {
        const SETTING: &str = "In a mysterious maze-like dungeon full of deadly traps and valuable treasure. A group of adventurers are exploring the dungeon.";
        const CHARACTERS: &[&str] = &["James", "Raven", "Morgan"];
        const SEED: u64 = 6532;

        // Create the model
        let model = Model::new(SEED, true).unwrap();

        // Create a scene
        let mut scene = model.create_scene(SETTING, CHARACTERS);

        // Display the setting and characters
        println!("Setting: {}", SETTING);
        println!("Characters: {:?}", CHARACTERS);

        // Infer a few turns
        for _ in 0..20 {
            // Infer a story turn
            let story_turn = scene.infer_any(50);

            // Display the the story turn
            match story_turn.turn_type() {
                scene::SceneTurnType::Story(story) => println!("[STORY] {}", story),
                scene::SceneTurnType::Dialogue(character, dialogue) => {
                    println!("[SAY] {}: \"{}\"", character, dialogue)
                }
            }
        }
    }

    #[test]
    fn crafting() {
        const SEED: u64 = 6532;

        // Create the model
        let model = Model::new(SEED, true).unwrap();

        // Create a crafter
        let crafter = Crafter::new(
            model,
            &[
                CrafterExample::new(&["water", "fire"], "steam"),
                CrafterExample::new(&["sugar", "water", "bee"], "honey"),
                CrafterExample::new(&["weapon", "life"], "death"),
                CrafterExample::new(&["light", "electricity"], "lightbulb"),
                CrafterExample::new(&["bird", "stick", "stick"], "nest"),
                CrafterExample::new(&["human", "hammer"], "construction worker"),
                CrafterExample::new(&["staff", "book"], "grimoire"),
            ],
        );

        // Craft some items
        let result = crafter.craft(&["water", "cloud"]);
        println!("water + cloud = {}", result);
        let result = crafter.craft(&["dragon", "wizard"]);
        println!("dragon + wizard = {}", result);
        let result = crafter.craft(&["demons", "angels"]);
        println!("demons + angels = {}", result);
        let result = crafter.craft(&["politics", "sword", "bomb"]);
        println!("politics + sword + bomb = {}", result);
        let result = crafter.craft(&["war", "tea"]);
        println!("war + tea = {}", result);
    }
}
