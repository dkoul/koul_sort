use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const MIN_RUN: usize = 32;
const RADIX_BITS: usize = 8;
const RADIX_SIZE: usize = 1 << RADIX_BITS; // 256 buckets

/// Compute minimum run length for efficient merging
fn compute_min_run(mut n: usize) -> usize {
    let mut r = 0;
    while n >= MIN_RUN {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

/// In-place insertion sort on a slice
fn insertion_sort<T: Ord + Copy>(arr: &mut [T], start: usize, end: usize) {
    for i in (start + 1)..end {
        let key = arr[i];
        let mut j = i;
        while j > start && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

/// Find a natural run starting at 'start'
/// Returns (end, is_descending)
fn find_run<T: Ord>(arr: &[T], start: usize) -> (usize, bool) {
    let n = arr.len();
    if start >= n - 1 {
        return (n, false);
    }

    let mut end = start + 1;

    if arr[end] < arr[start] {
        // Descending run
        while end < n && arr[end] < arr[end - 1] {
            end += 1;
        }
        (end, true)
    } else {
        // Ascending run (includes equal elements for stability)
        while end < n && arr[end] >= arr[end - 1] {
            end += 1;
        }
        (end, false)
    }
}

/// Reverse a slice in-place
fn reverse<T>(arr: &mut [T], start: usize, end: usize) {
    let mut i = start;
    let mut j = end - 1;
    while i < j {
        arr.swap(i, j);
        i += 1;
        j -= 1;
    }
}

/// Merge two adjacent sorted runs into the first position
/// Uses a temporary buffer for efficiency
fn merge<T: Ord + Copy + Default>(arr: &mut [T], base1: usize, len1: usize, len2: usize, buffer: &mut Vec<T>) {
    // Copy first run to buffer
    buffer.clear();
    buffer.extend_from_slice(&arr[base1..base1 + len1]);

    let mut cursor1 = 0;
    let mut cursor2 = base1 + len1;
    let mut dest = base1;
    let end2 = base1 + len1 + len2;

    while cursor1 < len1 && cursor2 < end2 {
        if buffer[cursor1] <= arr[cursor2] {
            arr[dest] = buffer[cursor1];
            cursor1 += 1;
        } else {
            arr[dest] = arr[cursor2];
            cursor2 += 1;
        }
        dest += 1;
    }

    // Copy remaining from buffer
    while cursor1 < len1 {
        arr[dest] = buffer[cursor1];
        cursor1 += 1;
        dest += 1;
    }
}

/// Run descriptor
struct Run {
    start: usize,
    len: usize,
}

/// Merge at stack position i with i+1
fn merge_at<T: Ord + Copy + Default>(arr: &mut [T], stack: &mut Vec<Run>, i: usize, buffer: &mut Vec<T>) {
    let base1 = stack[i].start;
    let len1 = stack[i].len;
    let len2 = stack[i + 1].len;

    merge(arr, base1, len1, len2, buffer);

    stack[i].len = len1 + len2;
    stack.remove(i + 1);
}

/// Maintain Timsort stack invariants
fn merge_collapse<T: Ord + Copy + Default>(arr: &mut [T], stack: &mut Vec<Run>, buffer: &mut Vec<T>) {
    while stack.len() > 1 {
        let n = stack.len() - 1;

        if n >= 2 && stack[n - 2].len <= stack[n - 1].len + stack[n].len {
            if stack[n - 2].len < stack[n].len {
                merge_at(arr, stack, n - 2, buffer);
            } else {
                merge_at(arr, stack, n - 1, buffer);
            }
        } else if stack[n - 1].len <= stack[n].len {
            merge_at(arr, stack, n - 1, buffer);
        } else {
            break;
        }
    }
}

/// Force merge all remaining runs
fn merge_force_collapse<T: Ord + Copy + Default>(arr: &mut [T], stack: &mut Vec<Run>, buffer: &mut Vec<T>) {
    while stack.len() > 1 {
        let n = stack.len() - 1;
        if n >= 2 && stack[n - 2].len < stack[n].len {
            merge_at(arr, stack, n - 2, buffer);
        } else {
            merge_at(arr, stack, n - 1, buffer);
        }
    }
}

/// KoulSort with natural run detection (Timsort-style)
pub fn koul_sort<T: Ord + Copy + Default>(arr: &mut [T]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // For small arrays, just use insertion sort
    if n < MIN_RUN {
        insertion_sort(arr, 0, n);
        return;
    }

    let min_run = compute_min_run(n);
    let mut stack: Vec<Run> = Vec::with_capacity(40); // log2(n) is enough
    let mut buffer: Vec<T> = Vec::with_capacity(n / 2);

    let mut start = 0;
    while start < n {
        // Find natural run
        let (mut end, descending) = find_run(arr, start);

        // Reverse if descending
        if descending {
            reverse(arr, start, end);
        }

        // Extend short runs with insertion sort
        let run_len = end - start;
        if run_len < min_run {
            let force = std::cmp::min(min_run, n - start);
            insertion_sort(arr, start, start + force);
            end = start + force;
        }

        // Push run onto stack
        stack.push(Run {
            start,
            len: end - start,
        });

        // Maintain merge invariants
        merge_collapse(arr, &mut stack, &mut buffer);

        start = end;
    }

    // Merge remaining runs
    merge_force_collapse(arr, &mut stack, &mut buffer);
}

/// Hybrid approach: detect characteristics per chunk, choose algorithm
pub fn koul_sort_hybrid<T: Ord + Copy + Default>(arr: &mut [T]) {
    let n = arr.len();
    if n < 64 {
        koul_sort(arr);
        return;
    }

    // Use chunk-based approach with sqrt(n) chunks
    let chunk_size = (n as f64).sqrt() as usize;
    let mut sorted_chunks: Vec<Vec<T>> = Vec::new();

    for chunk in arr.chunks(chunk_size) {
        let mut chunk_vec: Vec<T> = chunk.to_vec();
        koul_sort(&mut chunk_vec);
        sorted_chunks.push(chunk_vec);
    }

    // K-way merge using iterators
    let mut result: Vec<T> = Vec::with_capacity(n);
    let mut indices: Vec<usize> = vec![0; sorted_chunks.len()];

    for _ in 0..n {
        let mut min_val: Option<T> = None;
        let mut min_idx = 0;

        for (i, chunk) in sorted_chunks.iter().enumerate() {
            if indices[i] < chunk.len() {
                let val = chunk[indices[i]];
                if min_val.is_none() || val < min_val.unwrap() {
                    min_val = Some(val);
                    min_idx = i;
                }
            }
        }

        if let Some(val) = min_val {
            result.push(val);
            indices[min_idx] += 1;
        }
    }

    arr.copy_from_slice(&result);
}

/// Check if array is nearly sorted (< 10% inversions)
fn is_nearly_sorted<T: Ord>(arr: &[T]) -> bool {
    if arr.len() < 2 {
        return true;
    }
    let mut inversions = 0;
    for i in 0..arr.len() - 1 {
        if arr[i] > arr[i + 1] {
            inversions += 1;
        }
    }
    inversions < arr.len() / 10
}

/// Count natural runs in an array (for deciding algorithm)
fn count_runs<T: Ord>(arr: &[T]) -> usize {
    if arr.len() < 2 {
        return 1;
    }
    let mut runs = 1;
    let mut ascending = arr[1] >= arr[0];
    for i in 2..arr.len() {
        let new_ascending = arr[i] >= arr[i - 1];
        if new_ascending != ascending {
            runs += 1;
            ascending = new_ascending;
        }
    }
    runs
}

// ==================
// Counting Sort for dense ranges
// ==================

/// Check if counting sort is appropriate
/// Returns Some((min, max)) if range is small enough, None otherwise
fn counting_sort_viable(arr: &[i64]) -> Option<(i64, i64)> {
    if arr.len() < 64 {
        return None;
    }

    let mut min_val = arr[0];
    let mut max_val = arr[0];

    for &val in arr.iter() {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    // Calculate range safely to avoid overflow
    // Use u128 for the calculation to handle extreme cases
    let range = (max_val as i128 - min_val as i128) as u128;
    let threshold = (arr.len() * 2) as u128;

    // Use counting sort if range <= 2 * n (dense enough)
    // This ensures O(n) time and reasonable memory
    if range <= threshold {
        Some((min_val, max_val))
    } else {
        None
    }
}

/// Counting sort for integers with small range
/// O(n + k) where k is the range of values
pub fn counting_sort_i64(arr: &mut [i64], min_val: i64, max_val: i64) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    let range = (max_val - min_val + 1) as usize;
    let mut counts: Vec<usize> = vec![0; range];

    // Count occurrences
    for &val in arr.iter() {
        counts[(val - min_val) as usize] += 1;
    }

    // Reconstruct sorted array
    let mut idx = 0;
    for (val_offset, &count) in counts.iter().enumerate() {
        let val = min_val + val_offset as i64;
        for _ in 0..count {
            arr[idx] = val;
            idx += 1;
        }
    }
}

// ==================
// Radix Sort for i64
// ==================

/// LSD Radix sort for signed 64-bit integers
/// Uses XOR trick to handle negative numbers correctly
pub fn radix_sort_i64(arr: &mut [i64]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    // For small arrays, use insertion sort
    if n < 64 {
        insertion_sort(arr, 0, n);
        return;
    }

    let mut buffer: Vec<i64> = vec![0; n];
    let mut counts: [usize; RADIX_SIZE] = [0; RADIX_SIZE];

    // Process 8 bytes (64 bits) in 8 passes of 8 bits each
    for pass in 0..8 {
        let shift = pass * RADIX_BITS;

        // Count occurrences
        counts.fill(0);
        for &val in arr.iter() {
            // XOR with sign bit to handle negative numbers
            // This makes negative numbers sort before positive
            let transformed = (val as u64) ^ (1u64 << 63);
            let bucket = ((transformed >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        // Convert counts to prefix sums (starting positions)
        let mut total = 0;
        for count in counts.iter_mut() {
            let old_count = *count;
            *count = total;
            total += old_count;
        }

        // Place elements into buffer
        for &val in arr.iter() {
            let transformed = (val as u64) ^ (1u64 << 63);
            let bucket = ((transformed >> shift) & 0xFF) as usize;
            buffer[counts[bucket]] = val;
            counts[bucket] += 1;
        }

        // Swap arrays
        arr.copy_from_slice(&buffer);
    }
}

/// Radix sort for unsigned 64-bit integers (simpler, no sign handling)
pub fn radix_sort_u64(arr: &mut [u64]) {
    let n = arr.len();
    if n < 2 {
        return;
    }

    if n < 64 {
        arr.sort_unstable();
        return;
    }

    let mut buffer: Vec<u64> = vec![0; n];
    let mut counts: [usize; RADIX_SIZE] = [0; RADIX_SIZE];

    for pass in 0..8 {
        let shift = pass * RADIX_BITS;

        counts.fill(0);
        for &val in arr.iter() {
            let bucket = ((val >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        let mut total = 0;
        for count in counts.iter_mut() {
            let old_count = *count;
            *count = total;
            total += old_count;
        }

        for &val in arr.iter() {
            let bucket = ((val >> shift) & 0xFF) as usize;
            buffer[counts[bucket]] = val;
            counts[bucket] += 1;
        }

        arr.copy_from_slice(&buffer);
    }
}

// ==================
// Full Adaptive Sort
// ==================

/// KoulSort: Full adaptive sort that picks the best algorithm
///
/// Decision tree:
/// 1. n < 64: insertion sort
/// 2. Check if counting sort viable (small value range): counting sort O(n+k)
/// 3. Count natural runs in O(n) scan:
///    - runs <= 2: run-detection sort (nearly sorted or reverse)
///    - runs < sqrt(n): run-detection sort (structured data)
///    - runs >= sqrt(n): radix sort (random data)
pub fn koul_sort_v3(arr: &mut [i64]) {
    let n = arr.len();

    if n < 2 {
        return;
    }

    if n < 64 {
        insertion_sort(arr, 0, n);
        return;
    }

    // Check if counting sort is viable (small range = many duplicates)
    if let Some((min_val, max_val)) = counting_sort_viable(arr) {
        counting_sort_i64(arr, min_val, max_val);
        return;
    }

    // Quick O(n) scan to count natural runs
    let runs = count_runs(arr);
    let sqrt_n = (n as f64).sqrt() as usize;

    if runs <= 2 {
        // Very structured: 1-2 natural runs
        // Run detection will be O(n)
        koul_sort(arr);
    } else if runs < sqrt_n {
        // Moderately structured: use run detection
        koul_sort(arr);
    } else {
        // Random data: use radix sort O(n)
        radix_sort_i64(arr);
    }
}

/// Adaptive sort that picks strategy based on data
pub fn koul_sort_adaptive(arr: &mut [i64]) {
    koul_sort_v3(arr);
}

// ==================
// Python bindings
// ==================

#[pyfunction]
fn sort_i64<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    koul_sort(&mut data);
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn sort_f64<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> {
    let mut data: Vec<f64> = arr.as_slice().unwrap().to_vec();
    // For floats, we need to handle NaN - use total_cmp
    data.sort_by(|a, b| a.total_cmp(b));
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn sort_hybrid_i64<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    koul_sort_hybrid(&mut data);
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn sort_adaptive_i64<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    koul_sort_adaptive(&mut data);
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn rust_std_sort_i64<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    data.sort_unstable();
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn radix_sort<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    radix_sort_i64(&mut data);
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn sort_v3<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    koul_sort_v3(&mut data);
    PyArray1::from_vec_bound(py, data)
}

#[pyfunction]
fn counting_sort<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, i64>) -> Bound<'py, PyArray1<i64>> {
    let mut data: Vec<i64> = arr.as_slice().unwrap().to_vec();
    if let Some((min_val, max_val)) = counting_sort_viable(&data) {
        counting_sort_i64(&mut data, min_val, max_val);
    } else {
        // Fallback to radix if not viable
        radix_sort_i64(&mut data);
    }
    PyArray1::from_vec_bound(py, data)
}

#[pymodule]
#[pyo3(name = "koul_sort")]
fn koul_sort_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sort_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_f64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_hybrid_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sort_adaptive_i64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_std_sort_i64, m)?)?;
    m.add_function(wrap_pyfunction!(radix_sort, m)?)?;
    m.add_function(wrap_pyfunction!(sort_v3, m)?)?;
    m.add_function(wrap_pyfunction!(counting_sort, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let mut arr: Vec<i64> = vec![];
        koul_sort(&mut arr);
        assert_eq!(arr, vec![]);
    }

    #[test]
    fn test_single() {
        let mut arr = vec![42];
        koul_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }

    #[test]
    fn test_sorted() {
        let mut arr: Vec<i64> = (0..100).collect();
        let expected: Vec<i64> = (0..100).collect();
        koul_sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_reverse() {
        let mut arr: Vec<i64> = (0..100).rev().collect();
        let expected: Vec<i64> = (0..100).collect();
        koul_sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_random() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        koul_sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_duplicates() {
        let mut arr = vec![5, 5, 5, 5, 5];
        koul_sort(&mut arr);
        assert_eq!(arr, vec![5, 5, 5, 5, 5]);
    }

    #[test]
    fn test_large() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut arr: Vec<i64> = (0..10000)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                h.finish() as i64
            })
            .collect();
        let mut expected = arr.clone();
        expected.sort();
        koul_sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_radix_basic() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        radix_sort_i64(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_radix_negative() {
        let mut arr = vec![-5, 3, -1, 0, 2, -10, 7, -3];
        let mut expected = arr.clone();
        expected.sort();
        radix_sort_i64(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_radix_large() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut arr: Vec<i64> = (0..10000)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                // Use wrapping arithmetic to get mix of positive and negative
                (h.finish() as i64).wrapping_sub(i64::MAX / 2)
            })
            .collect();
        let mut expected = arr.clone();
        expected.sort();
        radix_sort_i64(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_v3_sorted() {
        let mut arr: Vec<i64> = (0..1000).collect();
        let expected: Vec<i64> = (0..1000).collect();
        koul_sort_v3(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_v3_random() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut arr: Vec<i64> = (0..10000)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                h.finish() as i64
            })
            .collect();
        let mut expected = arr.clone();
        expected.sort();
        koul_sort_v3(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_counting_basic() {
        // Need >= 64 elements for counting_sort_viable to return Some
        let mut arr: Vec<i64> = (0..100).map(|i| i % 10).collect();
        let mut expected = arr.clone();
        expected.sort();
        assert!(counting_sort_viable(&arr).is_some());
        if let Some((min, max)) = counting_sort_viable(&arr) {
            counting_sort_i64(&mut arr, min, max);
        }
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_counting_duplicates() {
        // Many duplicates in small range - ideal for counting sort
        let mut arr: Vec<i64> = (0..10000).map(|i| i % 100).collect();
        let mut expected = arr.clone();
        expected.sort();

        assert!(counting_sort_viable(&arr).is_some());
        if let Some((min, max)) = counting_sort_viable(&arr) {
            counting_sort_i64(&mut arr, min, max);
        }
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_counting_negative() {
        // Need >= 64 elements, with negative values
        let mut arr: Vec<i64> = (0..200).map(|i| (i % 20) - 10).collect();
        let mut expected = arr.clone();
        expected.sort();
        assert!(counting_sort_viable(&arr).is_some());
        if let Some((min, max)) = counting_sort_viable(&arr) {
            counting_sort_i64(&mut arr, min, max);
        }
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_v3_duplicates() {
        // This should trigger counting sort
        let mut arr: Vec<i64> = (0..10000).map(|i| i % 50).collect();
        let mut expected = arr.clone();
        expected.sort();
        koul_sort_v3(&mut arr);
        assert_eq!(arr, expected);
    }
}
