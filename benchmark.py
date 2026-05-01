import numpy as np
import time
import koul_sort

def benchmark(func, arr, runs=10):
    """Run a sort function multiple times and return average time in ms."""
    times = []
    for _ in range(runs):
        arr_copy = arr.copy()
        start = time.perf_counter()
        func(arr_copy)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times)


def count_runs(arr):
    """Count natural runs for analysis."""
    if len(arr) < 2:
        return 1
    runs = 1
    ascending = arr[1] >= arr[0]
    for i in range(2, len(arr)):
        new_ascending = arr[i] >= arr[i-1]
        if new_ascending != ascending:
            runs += 1
            ascending = new_ascending
    return runs


def nearly_sorted(n):
    arr = np.arange(n, dtype=np.int64)
    swaps = n // 20
    for _ in range(swaps):
        i, j = np.random.randint(0, n, 2)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def few_runs(n, num_runs):
    run_size = n // num_runs
    runs = []
    for i in range(num_runs):
        if i % 2 == 0:
            runs.append(np.arange(run_size, dtype=np.int64))
        else:
            runs.append(np.arange(run_size, 0, -1, dtype=np.int64))
    return np.concatenate(runs)[:n]


def run_benchmarks():
    print("=" * 130)
    print("RUST KoulSort v3 BENCHMARKS (Adaptive: Run-detection + Radix + Counting)")
    print("=" * 130)
    print()
    print(f"{'Scenario':<16} {'Size':>6} {'koul':>9} {'np.quick':>9} {'np.merge':>9} {'np.heap':>9} {'np.default':>10} {'rust_std':>9}")
    print("-" * 130)

    sizes = [10_000, 100_000, 1_000_000, 10_000_000]

    scenarios = {
        "random": lambda n: np.random.randint(0, n * 10, n, dtype=np.int64),
        "sorted": lambda n: np.arange(n, dtype=np.int64),
        "reverse": lambda n: np.arange(n, 0, -1, dtype=np.int64),
        "nearly sorted": nearly_sorted,
        "few runs (10)": lambda n: few_runs(n, 10),
        "many runs (100)": lambda n: few_runs(n, 100),
        "pipe organ": lambda n: np.concatenate([
            np.arange(n//2, dtype=np.int64),
            np.arange(n//2, 0, -1, dtype=np.int64)
        ]),
        "duplicates": lambda n: np.random.randint(0, 100, n, dtype=np.int64),
        "negative mix": lambda n: np.random.randint(-n, n, n, dtype=np.int64),
    }

    for scenario_name, generator in scenarios.items():
        for size in sizes:
            arr = generator(size)

            # KoulSort v3
            v3_time = benchmark(koul_sort.sort_v3, arr)

            # NumPy sorts
            np_quick = benchmark(lambda a: np.sort(a, kind='quicksort'), arr)
            np_merge = benchmark(lambda a: np.sort(a, kind='mergesort'), arr)
            np_heap = benchmark(lambda a: np.sort(a, kind='heapsort'), arr)
            np_default = benchmark(np.sort, arr)

            # Rust std sort
            std_time = benchmark(koul_sort.rust_std_sort_i64, arr)

            size_str = f"{size//1000}K" if size < 1_000_000 else f"{size//1_000_000}M"

            print(f"{scenario_name:<16} {size_str:>6} {v3_time:>7.2f}ms {np_quick:>7.2f}ms {np_merge:>7.2f}ms {np_heap:>7.2f}ms {np_default:>8.2f}ms {std_time:>7.2f}ms")
        print()


def run_summary():
    """Run a focused comparison at 10M elements."""
    print("\n" + "=" * 100)
    print("SUMMARY: 10M elements - KoulSort v3 vs All Competitors")
    print("=" * 100)
    print()
    print(f"{'Scenario':<16} {'koul':>10} {'np.quick':>10} {'np.merge':>10} {'np.heap':>10} {'np.default':>10} {'Winner':>12}")
    print("-" * 100)

    size = 10_000_000

    scenarios = {
        "random": lambda n: np.random.randint(0, n * 10, n, dtype=np.int64),
        "sorted": lambda n: np.arange(n, dtype=np.int64),
        "reverse": lambda n: np.arange(n, 0, -1, dtype=np.int64),
        "nearly sorted": nearly_sorted,
        "few runs (10)": lambda n: few_runs(n, 10),
        "pipe organ": lambda n: np.concatenate([
            np.arange(n//2, dtype=np.int64),
            np.arange(n//2, 0, -1, dtype=np.int64)
        ]),
        "duplicates": lambda n: np.random.randint(0, 100, n, dtype=np.int64),
    }

    pulse_wins = 0
    total = 0

    for scenario_name, generator in scenarios.items():
        arr = generator(size)

        v3_time = benchmark(koul_sort.sort_v3, arr)
        np_quick = benchmark(lambda a: np.sort(a, kind='quicksort'), arr)
        np_merge = benchmark(lambda a: np.sort(a, kind='mergesort'), arr)
        np_heap = benchmark(lambda a: np.sort(a, kind='heapsort'), arr)
        np_default = benchmark(np.sort, arr)

        times = {
            'koul': v3_time,
            'np.quick': np_quick,
            'np.merge': np_merge,
            'np.heap': np_heap,
            'np.default': np_default,
        }
        winner = min(times, key=times.get)
        if winner == 'koul':
            pulse_wins += 1
            winner_str = f"✓ {winner}"
        else:
            winner_str = winner
        total += 1

        print(f"{scenario_name:<16} {v3_time:>8.1f}ms {np_quick:>8.1f}ms {np_merge:>8.1f}ms {np_heap:>8.1f}ms {np_default:>8.1f}ms {winner_str:>12}")

    print()
    print(f"KoulSort v3 wins: {pulse_wins}/{total} scenarios")


def test_correctness():
    """Verify Rust implementation is correct."""
    print("CORRECTNESS TESTS")
    print("-" * 40)

    test_cases = [
        ("empty", np.array([], dtype=np.int64)),
        ("single", np.array([42], dtype=np.int64)),
        ("sorted", np.arange(1000, dtype=np.int64)),
        ("reverse", np.arange(1000, 0, -1, dtype=np.int64)),
        ("random", np.random.randint(0, 10000, 10000, dtype=np.int64)),
        ("duplicates", np.random.randint(0, 10, 1000, dtype=np.int64)),
        ("pipe organ", np.concatenate([np.arange(500, dtype=np.int64), np.arange(500, 0, -1, dtype=np.int64)])),
    ]

    all_passed = True
    for name, arr in test_cases:
        expected = np.sort(arr.copy())
        result = koul_sort.sort_i64(arr.copy())

        if np.array_equal(result, expected):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False

    print()
    return all_passed


if __name__ == "__main__":
    if test_correctness():
        run_benchmarks()
        run_summary()
    else:
        print("Tests failed!")
