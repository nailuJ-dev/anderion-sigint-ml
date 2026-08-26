use anderion_sigint_ml::{ClassScore, Observation};
use proptest::prelude::*;

proptest! {
    #[test]
    fn finite_nonempty_observations_roundtrip(values in prop::collection::vec(-1.0e3f32..1.0e3f32, 1..256)) {
        let observation = Observation::new("p", 0, values.clone()).unwrap();
        prop_assert_eq!(observation.features(), values.as_slice());
    }

    #[test]
    fn valid_probabilities_are_accepted(p in 0.0f32..=1.0f32) {
        let score = ClassScore::new("x", p).unwrap();
        prop_assert!((0.0..=1.0).contains(&score.probability()));
    }
}
