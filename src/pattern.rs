use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{ConceptKind, Prediction, Result, SdkError};

const DEFAULT_MAX_EVENTS: usize = 1_000_000;
const DEFAULT_MAX_SEQUENCE_LEN: usize = 16;
const MAX_SYMBOL_BYTES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PatternToken {
    concept: ConceptKind,
    symbol: String,
    embedding_cluster: Option<String>,
}

impl PatternToken {
    pub fn new(
        concept: ConceptKind,
        symbol: impl Into<String>,
        embedding_cluster: Option<String>,
    ) -> Result<Self> {
        let symbol = symbol.into();
        validate_token_text("pattern symbol", &symbol)?;
        if let Some(cluster) = &embedding_cluster {
            validate_token_text("embedding cluster", cluster)?;
        }
        Ok(Self {
            concept,
            symbol,
            embedding_cluster,
        })
    }

    pub fn concept(&self) -> ConceptKind {
        self.concept
    }
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
    pub fn embedding_cluster(&self) -> Option<&str> {
        self.embedding_cluster.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternEvent {
    timestamp_ms: u64,
    token: PatternToken,
}

impl PatternEvent {
    pub fn new(timestamp_ms: u64, token: PatternToken) -> Self {
        Self {
            timestamp_ms,
            token,
        }
    }
    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
    pub fn token(&self) -> &PatternToken {
        &self.token
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurringPattern {
    tokens: Vec<PatternToken>,
    occurrences: usize,
    occurrence_starts_ms: Vec<u64>,
}

impl RecurringPattern {
    pub fn tokens(&self) -> &[PatternToken] {
        &self.tokens
    }
    pub fn occurrences(&self) -> usize {
        self.occurrences
    }
    pub fn occurrence_starts_ms(&self) -> &[u64] {
        &self.occurrence_starts_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CooccurrencePattern {
    left: PatternToken,
    right: PatternToken,
    occurrences: usize,
}

impl CooccurrencePattern {
    pub fn left(&self) -> &PatternToken {
        &self.left
    }
    pub fn right(&self) -> &PatternToken {
        &self.right
    }
    pub fn occurrences(&self) -> usize {
        self.occurrences
    }
}

pub fn pattern_event_from_prediction(
    timestamp_ms: u64,
    prediction: &Prediction,
    embedding_cluster: Option<String>,
) -> Result<PatternEvent> {
    let top = prediction
        .top()
        .ok_or_else(|| SdkError::InvalidArgument("prediction has no top class".into()))?;
    Ok(PatternEvent::new(
        timestamp_ms,
        PatternToken::new(
            ConceptKind::SignalEvent,
            top.label().to_string(),
            embedding_cluster,
        )?,
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternEngine {
    max_events: usize,
    max_sequence_len: usize,
}

impl Default for PatternEngine {
    fn default() -> Self {
        Self {
            max_events: DEFAULT_MAX_EVENTS,
            max_sequence_len: DEFAULT_MAX_SEQUENCE_LEN,
        }
    }
}

impl PatternEngine {
    pub fn with_limits(max_events: usize, max_sequence_len: usize) -> Result<Self> {
        if max_events == 0 || !(2..=DEFAULT_MAX_SEQUENCE_LEN).contains(&max_sequence_len) {
            return Err(SdkError::InvalidArgument(
                "invalid pattern engine limits".into(),
            ));
        }
        Ok(Self {
            max_events,
            max_sequence_len,
        })
    }

    pub fn detect_sequences(
        &self,
        events: &[PatternEvent],
        sequence_len: usize,
        min_occurrences: usize,
    ) -> Result<Vec<RecurringPattern>> {
        self.validate_request(events, min_occurrences)?;
        if !(2..=self.max_sequence_len).contains(&sequence_len) {
            return Err(SdkError::InvalidArgument(
                "sequence_len is outside configured limits".into(),
            ));
        }
        if events.len() < sequence_len {
            return Ok(Vec::new());
        }
        let ordered = ordered_events(events);
        let mut counts: BTreeMap<Vec<PatternToken>, Vec<u64>> = BTreeMap::new();
        for window in ordered.windows(sequence_len) {
            let key: Vec<PatternToken> = window.iter().map(|event| event.token.clone()).collect();
            if let Some(first) = window.first() {
                counts.entry(key).or_default().push(first.timestamp_ms);
            }
        }
        let mut patterns: Vec<RecurringPattern> = counts
            .into_iter()
            .filter_map(|(tokens, starts)| {
                (starts.len() >= min_occurrences).then_some(RecurringPattern {
                    occurrences: starts.len(),
                    tokens,
                    occurrence_starts_ms: starts,
                })
            })
            .collect();
        patterns.sort_by(|a, b| {
            b.occurrences
                .cmp(&a.occurrences)
                .then_with(|| a.tokens.cmp(&b.tokens))
        });
        Ok(patterns)
    }

    pub fn detect_cooccurrences(
        &self,
        events: &[PatternEvent],
        bucket_ms: u64,
        min_occurrences: usize,
    ) -> Result<Vec<CooccurrencePattern>> {
        self.validate_request(events, min_occurrences)?;
        if bucket_ms == 0 {
            return Err(SdkError::InvalidArgument(
                "bucket_ms must be positive".into(),
            ));
        }
        let ordered = ordered_events(events);
        let mut buckets: BTreeMap<u64, BTreeSet<PatternToken>> = BTreeMap::new();
        for event in ordered {
            buckets
                .entry(event.timestamp_ms / bucket_ms)
                .or_default()
                .insert(event.token);
        }
        let mut counts: BTreeMap<(PatternToken, PatternToken), usize> = BTreeMap::new();
        for tokens in buckets.into_values() {
            let values: Vec<PatternToken> = tokens.into_iter().collect();
            for left_idx in 0..values.len() {
                for right_idx in left_idx.saturating_add(1)..values.len() {
                    let key = (values[left_idx].clone(), values[right_idx].clone());
                    *counts.entry(key).or_insert(0) += 1;
                }
            }
        }
        let mut patterns: Vec<CooccurrencePattern> = counts
            .into_iter()
            .filter_map(|((left, right), occurrences)| {
                (occurrences >= min_occurrences).then_some(CooccurrencePattern {
                    left,
                    right,
                    occurrences,
                })
            })
            .collect();
        patterns.sort_by(|a, b| {
            b.occurrences
                .cmp(&a.occurrences)
                .then_with(|| a.left.cmp(&b.left))
                .then_with(|| a.right.cmp(&b.right))
        });
        Ok(patterns)
    }

    fn validate_request(&self, events: &[PatternEvent], min_occurrences: usize) -> Result<()> {
        if events.len() > self.max_events {
            return Err(SdkError::DimensionLimit {
                actual: events.len(),
                max: self.max_events,
            });
        }
        if min_occurrences < 2 {
            return Err(SdkError::InvalidArgument(
                "min_occurrences must be at least 2".into(),
            ));
        }
        for event in events {
            validate_token_text("pattern symbol", event.token.symbol())?;
            if let Some(cluster) = event.token.embedding_cluster() {
                validate_token_text("embedding cluster", cluster)?;
            }
        }
        Ok(())
    }
}

fn ordered_events(events: &[PatternEvent]) -> Vec<PatternEvent> {
    let mut ordered = events.to_vec();
    ordered.sort_by(|a, b| {
        a.timestamp_ms
            .cmp(&b.timestamp_ms)
            .then_with(|| a.token.cmp(&b.token))
    });
    ordered
}

fn validate_token_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(SdkError::InvalidArgument(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_SYMBOL_BYTES {
        return Err(SdkError::DimensionLimit {
            actual: value.len(),
            max: MAX_SYMBOL_BYTES,
        });
    }
    Ok(())
}
