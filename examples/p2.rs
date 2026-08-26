use std::sync::Arc;

use anderion_sigint_ml::{
    FoundationModel, FoundationPooler, HashProjectionEncoder, SymmetricQuantizer,
    SyntheticDataAdapter, SyntheticFeatureGenerator, TemporalTransformerEncoder,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let generator = SyntheticFeatureGenerator::new(8, 42)?;
    let observations = generator.generate(4)?;

    let encoder = Arc::new(HashProjectionEncoder::new(8, 4, 7)?);
    let foundation = FoundationPooler::new(encoder, 16)?;
    let context = foundation.encode_window(&observations)?;

    let temporal = TemporalTransformerEncoder::new(1.0, 0.25)?;
    let contextualized = temporal.aggregate(&[context.clone(), context])?;

    let quantizer = SymmetricQuantizer::new(8)?;
    let compressed = quantizer.quantize(&contextualized)?;
    let restored = quantizer.dequantize(&compressed)?;

    println!("p2 embedding dimension: {}", restored.dim());
    Ok(())
}
