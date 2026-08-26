use std::sync::Arc;

use anderion_sigint_ml::{HashProjectionEncoder, Observation, Pipeline, PrototypeClassifier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = HashProjectionEncoder::new(4, 8, 42)?;
    let known_a = encoder.encode_features(&[1.0, 0.0, 0.1, 0.0])?;
    let known_b = encoder.encode_features(&[0.0, 1.0, 0.0, 0.1])?;
    let classifier = PrototypeClassifier::fit(&[
        (known_a, "class-a".to_string()),
        (known_b, "class-b".to_string()),
    ])?;
    let pipeline = Pipeline::new(Arc::new(encoder), Arc::new(classifier), 0.55)?;
    let observation = Observation::new("demo", 0, vec![0.9, 0.1, 0.1, 0.0])?;
    let prediction = pipeline.predict(&observation)?;
    println!("{}", serde_json::to_string_pretty(&prediction)?);
    Ok(())
}
