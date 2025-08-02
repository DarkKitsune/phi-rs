pub mod crafter;
pub mod model;
pub mod token_string;

#[cfg(test)]
mod tests {
    use super::*;
    use crafter::{Crafter, CrafterExample};
    use model::Model;

    #[test]
    fn crafting() {
        const SEED: u64 = 122534;

        // Create the model
        let model = Model::new(SEED, true).unwrap();

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
        let model = Model::new(SEED, true).unwrap();

        // Present choices to the model
        let item = model.try_choose_item(
            "You are a knight in a fantasy world.",
            "The item should be a weapon capable of defeating a dragon.",
            ["horse", "sword", "potion", "compass"],
            SEED,
            ATTEMPTS
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
}
