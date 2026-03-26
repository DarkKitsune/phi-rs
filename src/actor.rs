use std::fmt::Display;

pub struct Actor {
    name: String,
    identity: String,
}

impl Actor {
    pub fn new(name: impl Display, identity: impl Display) -> Self {
        Self {
            name: name.to_string(),
            identity: identity.to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}
