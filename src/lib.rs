pub mod crafter;
pub mod model;
pub mod scene;
pub mod token_string;

#[cfg(test)]
mod tests {
    use super::*;
    use crafter::{Crafter, CrafterExample};
    use model::Model;

    #[test]
    fn scene() {
        const SETTING: &str = "In a mysterious maze-like dungeon full of deadly traps and valuable treasure. A group of adventurers are exploring the dungeon.";
        const CHARACTERS: &[&str] = &["James", "Raven", "Jack", "Luna"];
        const SEED: u64 = 6789;

        // Create the model
        let model = Model::new(SEED, true).unwrap();

        // Create a scene
        let mut scene = model.create_scene(SETTING, CHARACTERS);

        // Display the setting and characters
        println!("Setting: {}", SETTING);
        println!("Characters: {:?}", CHARACTERS);

        // Infer enough turns to make memory compression happen
        for _ in 0..50 {
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

    #[test]
    fn pick_items() {
        const SEED: u64 = 545856;

        // Create the model
        let model = Model::new(SEED, true).unwrap();

        // Pick some items for various scenarios
        let scenario = "You are a knight in a fantasy world.";
        let desired_item_traits = "Something that will help you kill a dragon.";
        let available_items = &["horse", "sword", "potion", "compass"];
        let picked_item = model.pick_item(
            &model.tokenize(scenario),
            Some(&model.tokenize(desired_item_traits)),
            available_items,
            SEED
        ).unwrap();
        println!("Scenario: {}\nDesired item traits: {}\nAvailable items: {:?}\nPicked item: {}", scenario, desired_item_traits, available_items, picked_item);

        let scenario = "You are an explorer in a jungle.";
        let desired_item_traits = "Something that will help you cut through dense foliage.";
        let available_items = &["binoculars", "compass", "map", "machete", "book", "rope"];
        let picked_item = model.pick_item(
            &model.tokenize(scenario),
            Some(&model.tokenize(desired_item_traits)),
            available_items,
            SEED
        ).unwrap();
        println!("\nScenario: {}\nDesired item traits: {}\nAvailable items: {:?}\nPicked item: {}", scenario, desired_item_traits, available_items, picked_item);

        let scenario = "You are stranded on a deserted island.";
        let desired_item_traits = "Something that will help you start a fire.";
        let available_items = &["knife", "canteen", "matches", "lamp oil", "rope", "bombs"];
        let picked_item = model.pick_item(
            &model.tokenize(scenario),
            Some(&model.tokenize(desired_item_traits)),
            available_items,
            SEED
        ).unwrap();
        println!("\nScenario: {}\nDesired item traits: {}\nAvailable items: {:?}\nPicked item: {}", scenario, desired_item_traits, available_items, picked_item);
    }
}
