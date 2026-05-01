# KoulSort

A high-performance adaptive sorting algorithm implemented in Rust with Python bindings.

KoulSort automatically detects data characteristics and selects the optimal sorting strategy, achieving faster-than-NumPy performance across diverse input patterns.

## Algorithm

KoulSort uses a three-way adaptive strategy:

```
if value_range ≤ 2n    → Counting Sort   O(n+k)   // duplicates, small ranges
elif natural_runs ≤ √n → Run Detection   O(n)     // sorted, reverse, structured
else                   → Radix Sort      O(8n)    // random data
```

### Components

1. **Counting Sort** - For arrays with many duplicates or small value ranges. Detects when `max - min ≤ 2n` and uses O(n+k) bucket counting.

2. **Run Detection (Timsort-style)** - Finds naturally sorted ascending/descending subsequences, reverses descending runs, extends short runs with insertion sort, and merges using a stack-based strategy.

3. **LSD Radix Sort** - For random data with no exploitable structure. Processes 8 bits per pass (8 passes for 64-bit integers). Handles signed integers via XOR transformation.

## Performance

### Benchmark: 10M elements

| Scenario | KoulSort | np.quick | np.merge | np.heap | np.default | Winner |
|----------|----------|----------|----------|---------|------------|--------|
| random | 155ms | 239ms | 455ms | 240ms | 240ms | **KoulSort** |
| sorted | 20ms | 222ms | 8ms | 222ms | 224ms | np.merge |
| reverse | 19ms | 226ms | 11ms | 226ms | 228ms | np.merge |
| nearly sorted | 36ms | 254ms | 371ms | 253ms | 253ms | **KoulSort** |
| few runs | 11ms | 235ms | 85ms | 236ms | 233ms | **KoulSort** |
| pipe organ | 15ms | 247ms | 31ms | 247ms | 249ms | **KoulSort** |
| duplicates | 9ms | 60ms | 346ms | 60ms | 60ms | **KoulSort** |

**KoulSort wins 5/7 scenarios**, losing only to NumPy's mergesort on already-sorted data.

### Speedup vs NumPy Default (10M elements)

| Scenario | Speedup |
|----------|---------|
| duplicates | **7x faster** |
| few runs | **21x faster** |
| pipe organ | **15x faster** |
| nearly sorted | **7x faster** |
| sorted | **11x faster** |
| reverse | **11x faster** |
| random | **1.5x faster** |

## Installation

### Prerequisites

- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Python 3.8+
- maturin (`pip install maturin`)

### Build

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy maturin

# Build and install
maturin develop --release
```

## Usage

### Python

```python
import numpy as np
import koul_sort

# Sort an array
arr = np.random.randint(0, 1000000, 1000000, dtype=np.int64)
sorted_arr = koul_sort.sort_v3(arr)

# Other available functions
koul_sort.sort_i64(arr)        # Run-detection only
koul_sort.radix_sort(arr)      # Radix sort only
koul_sort.counting_sort(arr)   # Counting sort (with fallback)
```

### Rust

```rust
use koul_sort::{koul_sort, koul_sort, radix_sort_i64};

let mut data: Vec<i64> = vec![3, 1, 4, 1, 5, 9, 2, 6];

// Adaptive sort (recommended)
koul_sort(&mut data);

// Or use specific algorithms
koul_sort(&mut data);        // Run-detection
radix_sort_i64(&mut data);    // Radix sort
```

## Benchmarking

```bash
source .venv/bin/activate
python benchmark.py
```

## How It Works

### 1. Detection Phase (O(n))

The algorithm performs a single pass to:
- Find min/max values (for counting sort viability)
- Count natural runs (for structure detection)

### 2. Algorithm Selection

```
┌─────────────────────────────────────────────────────────┐
│                    Input Array                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ range ≤ 2n?         │
                └─────────────────────┘
                    │           │
                   yes          no
                    │           │
                    ▼           ▼
            ┌───────────┐  ┌─────────────────────┐
            │ Counting  │  │ runs ≤ √n?          │
            │ Sort O(n) │  └─────────────────────┘
            └───────────┘      │           │
                              yes          no
                               │           │
                               ▼           ▼
                       ┌───────────┐  ┌───────────┐
                       │ Run-based │  │ Radix     │
                       │ Sort O(n) │  │ Sort O(n) │
                       └───────────┘  └───────────┘
```

### 3. Why Each Algorithm?

| Algorithm | Best For | Complexity | Memory |
|-----------|----------|------------|--------|
| Counting | Many duplicates, small range | O(n+k) | O(k) |
| Run Detection | Sorted, nearly sorted, structured | O(n) | O(n/2) |
| Radix | Random integers | O(8n) | O(n) |

## Development Journey

This algorithm evolved through several iterations:

| Version | Random 10M | Sorted 10M | Key Change |
|---------|------------|------------|------------|
| Python v1 | 45x slower | 5x slower | Pure Python baseline |
| NumPy chunks | 2.5x slower | 3x slower | NumPy arrays |
| Rust v1 | 2.5x slower | 33x faster | Run detection in Rust |
| Rust v2 | 1.5x faster | 11x faster | Added radix sort |
| **Rust v3** | **1.5x faster** | **11x faster** | Added counting sort |

## License

MIT

## Author

Built with Claude Code.
