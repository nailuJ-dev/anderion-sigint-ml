use std::fs;

use anderion_sigint_ml::service::ReferenceModelBundle;
use anderion_sigint_ml::{ArtifactManifest, HashProjectionEncoder, PrototypeClassifier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = HashProjectionEncoder::new(4, 8, 42)?;
    let a = encoder.encode_features(&[1.0, 0.0, 0.1, 0.0])?;
    let b = encoder.encode_features(&[0.0, 1.0, 0.0, 0.1])?;
    let classifier =
        PrototypeClassifier::fit(&[(a, "class-a".to_string()), (b, "class-b".to_string())])?;
    let bundle = ReferenceModelBundle {
        encoder,
        classifier,
        unknown_threshold: 0.55,
    };
    let payload = serde_json::to_vec_pretty(&bundle)?;
    let manifest = ArtifactManifest::for_payload("demo-model", "reference_bundle", 1, &payload)?;
    fs::create_dir_all("artifacts")?;
    fs::write("artifacts/reference-model.json", &payload)?;
    fs::write(
        "artifacts/reference-model.manifest.json",
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}
