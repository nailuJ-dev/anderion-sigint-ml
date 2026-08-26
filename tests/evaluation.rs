use anderion_sigint_ml::{DatasetRow, classification_metrics, grouped_split};

#[test]
fn grouped_split_prevents_group_leakage() {
    let rows = vec![
        DatasetRow::new("1", "g1", "a", vec![1.0]).unwrap(),
        DatasetRow::new("2", "g1", "a", vec![1.1]).unwrap(),
        DatasetRow::new("3", "g2", "b", vec![2.0]).unwrap(),
        DatasetRow::new("4", "g3", "b", vec![2.1]).unwrap(),
    ];
    let split = grouped_split(&rows, 0.5, 7).unwrap();
    for train in &split.train {
        assert!(
            !split
                .test
                .iter()
                .any(|test| test.group_id == train.group_id)
        );
    }
}

#[test]
fn classification_metrics_are_correct() {
    let metrics = classification_metrics(&["a", "a", "b", "b"], &["a", "b", "b", "b"]).unwrap();
    assert!((metrics.accuracy - 0.75).abs() < 1e-6);
    assert!(metrics.macro_f1 > 0.7 && metrics.macro_f1 < 0.8);
}
