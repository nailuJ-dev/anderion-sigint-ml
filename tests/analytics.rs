use anderion_sigint_ml::{
    ClassScore, Embedding, SimilarityIndex, TemporalVote, WeightedEnsemble, kmeans,
};

#[test]
fn similarity_search_orders_nearest_first() {
    let mut index = SimilarityIndex::new(2).unwrap();
    index
        .insert("north", Embedding::new(vec![1.0, 0.0]).unwrap())
        .unwrap();
    index
        .insert("east", Embedding::new(vec![0.0, 1.0]).unwrap())
        .unwrap();
    let result = index
        .search(&Embedding::new(vec![0.9, 0.1]).unwrap(), 2)
        .unwrap();
    assert_eq!(result[0].id, "north");
}

#[test]
fn kmeans_separates_two_clusters() {
    let samples = vec![
        Embedding::new(vec![0.0, 0.0]).unwrap(),
        Embedding::new(vec![0.1, 0.0]).unwrap(),
        Embedding::new(vec![10.0, 10.0]).unwrap(),
        Embedding::new(vec![10.1, 10.0]).unwrap(),
    ];
    let clustered = kmeans(&samples, 2, 20).unwrap();
    assert_ne!(clustered.assignments[0], clustered.assignments[2]);
}

#[test]
fn temporal_vote_stabilizes_classification() {
    let vote = TemporalVote::new(0.8).unwrap();
    let history = vec![
        vec![
            ClassScore::new("a", 0.9).unwrap(),
            ClassScore::new("b", 0.1).unwrap(),
        ],
        vec![
            ClassScore::new("a", 0.8).unwrap(),
            ClassScore::new("b", 0.2).unwrap(),
        ],
        vec![
            ClassScore::new("a", 0.4).unwrap(),
            ClassScore::new("b", 0.6).unwrap(),
        ],
    ];
    let out = vote.aggregate(&history).unwrap();
    assert_eq!(out[0].label(), "a");
}

#[test]
fn weighted_ensemble_normalizes_output() {
    let ensemble = WeightedEnsemble::new(vec![0.75, 0.25]).unwrap();
    let out = ensemble
        .combine(&[
            vec![
                ClassScore::new("a", 0.9).unwrap(),
                ClassScore::new("b", 0.1).unwrap(),
            ],
            vec![
                ClassScore::new("a", 0.2).unwrap(),
                ClassScore::new("b", 0.8).unwrap(),
            ],
        ])
        .unwrap();
    let sum: f32 = out.iter().map(|s| s.probability()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert_eq!(out[0].label(), "a");
}
