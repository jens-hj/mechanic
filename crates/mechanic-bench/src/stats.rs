//! Percentiles, defined once so every report means the same thing by "p95".

/// Nearest-rank percentile of an ascending slice: the smallest sample with at
/// least `percentage` percent of the samples at or below it.
///
/// # Panics
///
/// Panics when `sorted` is empty.
pub fn percentile(sorted: &[f64], percentage: usize) -> f64 {
    let rank = sorted.len().saturating_mul(percentage).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

/// The 95th [`percentile`] of an ascending slice.
///
/// # Panics
///
/// Panics when `sorted` is empty.
pub fn percentile_95(sorted: &[f64]) -> f64 {
    percentile(sorted, 95)
}

/// The 95th percentile of unsorted samples, or zero when there are none.
pub fn percentile_95_or_zero(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        percentile_95(&sorted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_takes_the_sample_that_covers_the_requested_share() {
        let sorted: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((percentile(&sorted, 95) - 95.0).abs() < f64::EPSILON);
        assert!((percentile(&sorted, 50) - 50.0).abs() < f64::EPSILON);
        assert!((percentile(&[7.0], 95) - 7.0).abs() < f64::EPSILON);
        assert!(percentile_95_or_zero(&[]).abs() < f64::EPSILON);
    }
}
