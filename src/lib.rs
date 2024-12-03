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
        const SEED: u64 = 6532;

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
        let result = crafter.craft(&["evil", "good"], SEED);
        println!("demons + good = {}", result);
        let result = crafter.craft(&["politics", "sword", "bomb"], SEED);
        println!("politics + sword + bomb = {}", result);
        let result = crafter.craft(&["war", "a bottle of juice"], SEED);
        println!("war + a bottle of juice = {}", result);
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
            "You are going to a party. You want to bring a gift.",
            "A gift for an adult who loves animals and art.",
            ["vase", "book", "toy horse", "plant", "painting of a cat", "a calendar with pictures of birds"],
            SEED,
            ATTEMPTS
        );
        println!("Chose item: {:?}", item);

        let item = model.try_choose_item(
            "You are a grade school teacher. There is a new student in your class. You want to make them feel welcome.",
            "A gift for a child who likes to play outside.",
            ["puzzle", "ball", "coloring book", "stuffed animal"],
            SEED,
            ATTEMPTS
        );
        println!("Chose item: {:?}", item);
        
    }
}
