#!/usr/bin/env python3
"""
Example script demonstrating the usage of UpdateOperationTester class
for testing update operations (delete + re-add same elements) on SIFT dataset.
"""

from test_update_operations import UpdateOperationTester


def quick_demo():
    """Run a quick demonstration of update operations."""
    print("=== Quick Update Operations Demo ===\n")

    # Initialize tester with smaller parameters for quick testing
    tester = UpdateOperationTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        num_elements=1_000_000,   # Small dataset for quick testing
        num_queries=10_000,       # Queries sampled from training data
        k=10,                  # Top-10 nearest neighbors
        M=16,                  # HNSW parameter
        ef_construction=100,   # Lower for faster building
        ef_search=10,         # Lower for faster search
        seed=42
    )

    # Load data and build indices
    print("Loading data and building indices...")
    tester.load_data()
    tester.build_indices()

    # Test single update operation
    print("\n" + "="*50)
    print("SINGLE UPDATE OPERATION TEST")
    print("="*50)

    single_result = tester.test_single_update(update_ratio=0.1)

    print(f"\nSingle Update Results:")
    print(f"  Baseline recall: {single_result['baseline_recall']:.4f}")
    print(f"  After deletion: {single_result['after_deletion_recall']:.4f}")
    print(f"  After update: {single_result['after_update_recall']:.4f}")
    print(f"  Recall drop: {single_result['recall_drop_from_deletion']:.4f}")
    print(f"  Recall recovery: {single_result['recall_recovery']:.4f}")
    print(
        f"  Final difference: {single_result['final_recall_difference']:.4f}")
    print(f"  Total time: {single_result['total_time']:.3f}s")

    # Test multiple update operations
    print("\n" + "="*50)
    print("MULTIPLE UPDATE OPERATIONS TEST")
    print("="*50)

    multiple_results = tester.test_multiple_updates(
        update_ratio=0.1, num_iterations=3)

    # Plot results
    tester.plot_update_results(multiple_results, "quick_demo_results.png")

    return single_result, multiple_results


def parameter_study():
    """Study the effect of different update ratios."""
    print("\n=== Update Ratio Parameter Study ===\n")

    tester = UpdateOperationTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        num_elements=20_000,
        num_queries=1_000,
        k=10,
        M=16,
        ef_construction=150,
        ef_search=150,
        seed=123
    )

    # Load data and build indices
    tester.load_data()
    tester.build_indices()

    # Test different update ratios
    update_ratios = [0.05, 0.1, 0.2, 0.3]
    ratio_results = {}

    for ratio in update_ratios:
        print(f"\nTesting update ratio: {ratio:.1%}")
        result = tester.test_single_update(update_ratio=ratio)
        ratio_results[ratio] = result

        # Rebuild index for next test (to start fresh)
        tester.build_indices()

    # Print summary
    print(f"\n{'Ratio':<8} {'Baseline':<10} {'Drop':<10} {'Recovery':<10} {'Final':<10} {'Time':<8}")
    print("-" * 60)

    for ratio, result in ratio_results.items():
        print(f"{ratio:<8.1%} {result['baseline_recall']:<10.4f} "
              f"{result['recall_drop_from_deletion']:<10.4f} "
              f"{result['recall_recovery']:<10.4f} "
              f"{result['final_recall_difference']:<10.4f} "
              f"{result['total_time']:<8.3f}")

    return ratio_results


def stress_test():
    """Run a stress test with many iterations."""
    print("\n=== Stress Test: Many Update Iterations ===\n")

    tester = UpdateOperationTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        num_elements=15_000,
        num_queries=800,
        k=10,
        M=16,
        ef_construction=150,
        ef_search=150,
        seed=456
    )

    # Load data and build indices
    tester.load_data()
    tester.build_indices()

    # Run many iterations to see if recall degrades over time
    stress_results = tester.test_multiple_updates(
        update_ratio=0.15, num_iterations=10)

    # Analyze recall stability
    initial_recall = stress_results[0]['before_recall']
    final_recall = stress_results[-1]['after_update_recall']
    max_drop = max(r['recall_drop_from_deletion'] for r in stress_results)
    min_recovery = min(r['recall_recovery'] for r in stress_results)

    print(f"\nStress Test Analysis:")
    print(f"  Initial recall: {initial_recall:.4f}")
    print(f"  Final recall: {final_recall:.4f}")
    print(f"  Total change: {final_recall - initial_recall:.4f}")
    print(f"  Maximum recall drop: {max_drop:.4f}")
    print(f"  Minimum recall recovery: {min_recovery:.4f}")

    # Plot results
    tester.plot_update_results(stress_results, "stress_test_results.png")

    return stress_results


def comparison_test():
    """Compare different HNSW parameters."""
    print("\n=== Parameter Comparison Test ===\n")

    # Test different M values
    m_values = [8, 16, 32]
    comparison_results = {}

    for M in m_values:
        print(f"\nTesting M={M}")

        tester = UpdateOperationTester(
            data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
            num_elements=15_000,
            num_queries=500,
            k=10,
            M=M,
            ef_construction=100,
            ef_search=100,
            seed=123
        )

        # Load data and build indices
        tester.load_data()
        tester.build_indices()

        # Test update operation
        result = tester.test_single_update(update_ratio=0.1)
        comparison_results[M] = result

    # Print comparison
    print(f"\n{'M':<4} {'Baseline':<10} {'Drop':<10} {'Recovery':<10} {'Final':<10}")
    print("-" * 50)

    for M, result in comparison_results.items():
        print(f"{M:<4} {result['baseline_recall']:<10.4f} "
              f"{result['recall_drop_from_deletion']:<10.4f} "
              f"{result['recall_recovery']:<10.4f} "
              f"{result['final_recall_difference']:<10.4f}")

    return comparison_results


if __name__ == "__main__":
    # Run different tests

    # 1. Quick demonstration
    try:
        print("Running quick demo...")
        single_result, multiple_results = quick_demo()
    except Exception as e:
        print(f"Quick demo failed: {e}")

    # 2. Parameter study (uncomment to run)
    # try:
    #     print("Running parameter study...")
    #     ratio_results = parameter_study()
    # except Exception as e:
    #     print(f"Parameter study failed: {e}")

    # 3. Stress test (uncomment to run)
    # try:
    #     print("Running stress test...")
    #     stress_results = stress_test()
    # except Exception as e:
    #     print(f"Stress test failed: {e}")

    # 4. Comparison test (uncomment to run)
    # try:
    #     print("Running comparison test...")
    #     comparison_results = comparison_test()
    # except Exception as e:
    #     print(f"Comparison test failed: {e}")

    print("\n=== Tests completed ===")
