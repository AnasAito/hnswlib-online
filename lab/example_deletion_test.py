#!/usr/bin/env python3
"""
Example script demonstrating the usage of DeletionUpdateTester class
for testing deletion and update operations on SIFT dataset.
"""

from test_deletion_update import DeletionUpdateTester
import numpy as np


def quick_test():
    """Run a quick test with smaller parameters for demonstration."""
    print("=== Quick Deletion/Update Test Demo ===\n")

    # Initialize tester with smaller parameters for quick testing
    tester = DeletionUpdateTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        num_elements=10_000,   # Smaller dataset for quick testing
        num_queries=500,       # Fewer queries
        k=10,                  # Top-10 nearest neighbors
        M=16,                  # HNSW parameter
        ef_construction=100,   # Lower for faster building
        ef_search=100,         # Lower for faster search
        seed=42
    )

    # Load data and build indices
    print("Loading SIFT data...")
    tester.load_sift_data()

    print("Building indices...")
    tester.build_indices()

    # Test single deletion impact
    print("\n" + "="*50)
    print("SINGLE DELETION TEST")
    print("="*50)

    deletion_results = tester.test_deletion_impact(
        deletion_ratio=0.1,    # Delete 10% of elements
        num_iterations=2       # Run 2 iterations
    )

    # Test single replacement impact
    print("\n" + "="*50)
    print("SINGLE REPLACEMENT TEST")
    print("="*50)

    # Rebuild indices for replacement test
    tester.build_indices()

    replacement_results = tester.test_replacement_impact(
        replacement_ratio=0.1,  # Replace 10% of elements
        num_iterations=2        # Run 2 iterations
    )

    # Print results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)

    print("\nDeletion Results:")
    for i, result in enumerate(deletion_results):
        print(f"  Iteration {i+1}: "
              f"Baseline recall = {result['baseline_recall']:.4f}, "
              f"After deletion = {result['after_deletion_recall']:.4f}, "
              f"Drop = {result['recall_drop']:.4f}")

    print("\nReplacement Results:")
    for i, result in enumerate(replacement_results):
        print(f"  Iteration {i+1}: "
              f"Baseline recall = {result['baseline_recall']:.4f}, "
              f"After replacement = {result['after_replacement_recall']:.4f}, "
              f"Change = {result['recall_change']:.4f}")


def comprehensive_test():
    """Run comprehensive tests with multiple ratios."""
    print("\n=== Comprehensive Deletion/Update Test ===\n")

    # Initialize tester
    tester = DeletionUpdateTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        num_elements=25_000,   # Medium-sized dataset
        num_queries=1_000,
        k=10,
        M=16,
        ef_construction=150,
        ef_search=150,
        seed=123
    )

    # Run comprehensive tests
    results = tester.run_comprehensive_test(
        # Test different deletion ratios
        deletion_ratios=[0.05, 0.1, 0.15, 0.2],
        # Test different replacement ratios
        replacement_ratios=[0.05, 0.1, 0.15, 0.2],
        num_iterations=3                               # 3 iterations per test
    )

    # Print summary
    tester.print_summary()

    # Plot results
    tester.plot_results("comprehensive_test_results.png")

    return results


def parameter_sensitivity_test():
    """Test how different HNSW parameters affect deletion/replacement impact."""
    print("\n=== Parameter Sensitivity Test ===\n")

    # Test different M values
    m_values = [8, 16, 32]
    ef_values = [50, 100, 200]

    results_by_params = {}

    for M in m_values:
        for ef in ef_values:
            print(f"\nTesting M={M}, ef={ef}")

            tester = DeletionUpdateTester(
                data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
                num_elements=15_000,
                num_queries=500,
                k=10,
                M=M,
                ef_construction=ef,
                ef_search=ef,
                seed=123
            )

            # Load data and build indices
            tester.load_sift_data()
            tester.build_indices()

            # Test deletion impact
            deletion_results = tester.test_deletion_impact(
                deletion_ratio=0.1,
                num_iterations=1
            )

            # Store results
            key = f"M{M}_ef{ef}"
            results_by_params[key] = {
                'M': M,
                'ef': ef,
                'baseline_recall': deletion_results[0]['baseline_recall'],
                'recall_drop': deletion_results[0]['recall_drop'],
                'deletion_time': deletion_results[0]['deletion_time']
            }

    # Print parameter sensitivity results
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY RESULTS")
    print("="*60)
    print(f"{'Parameters':<15} {'Baseline':<10} {'Drop':<10} {'Time':<10}")
    print("-" * 45)

    for key, result in results_by_params.items():
        print(f"{key:<15} {result['baseline_recall']:<10.4f} "
              f"{result['recall_drop']:<10.4f} {result['deletion_time']:<10.3f}")

    return results_by_params


def custom_test():
    """Run a custom test with user-defined parameters."""
    print("\n=== Custom Test ===\n")

    # You can modify these parameters as needed
    custom_params = {
        'data_path': '/data/anas.aitaomar/sift_1m_old_dist.h5',
        'num_elements': 20_000,
        'num_queries': 800,
        'k': 15,
        'M': 20,
        'ef_construction': 200,
        'ef_search': 150,
        'seed': 456
    }

    print(f"Custom parameters: {custom_params}")

    tester = DeletionUpdateTester(**custom_params)

    # Load and build
    tester.load_sift_data()
    tester.build_indices()

    # Custom test: What happens when we delete many elements at once?
    high_deletion_results = tester.test_deletion_impact(
        deletion_ratio=0.3,  # Delete 30% of elements
        num_iterations=1
    )

    print(f"\nHigh deletion test results:")
    result = high_deletion_results[0]
    print(
        f"  Deleted: {result['num_deleted']} elements ({result['deletion_ratio']:.1%})")
    print(f"  Baseline recall: {result['baseline_recall']:.4f}")
    print(f"  After deletion recall: {result['after_deletion_recall']:.4f}")
    print(f"  Recall drop: {result['recall_drop']:.4f}")
    print(f"  Deletion time: {result['deletion_time']:.3f} seconds")

    return high_deletion_results


if __name__ == "__main__":
    # Run different types of tests

    # 1. Quick demonstration
    try:
        quick_test()
    except Exception as e:
        print(f"Quick test failed: {e}")

    # 2. Comprehensive test (comment out if too slow)
    # try:
    #     comprehensive_test()
    # except Exception as e:
    #     print(f"Comprehensive test failed: {e}")

    # 3. Parameter sensitivity test
    # try:
    #     parameter_sensitivity_test()
    # except Exception as e:
    #     print(f"Parameter sensitivity test failed: {e}")

    # 4. Custom test
    # try:
    #     custom_test()
    # except Exception as e:
    #     print(f"Custom test failed: {e}")

    print("\n=== All tests completed ===")
