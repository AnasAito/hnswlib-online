import hnswlib
import numpy as np
import h5py
import time
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt


class DeletionUpdateTester:
    """
    A class to test deletion and update operations on real datasets (like SIFT).

    This class provides methods to:
    1. Load real datasets (SIFT, YFCC, etc.) from h5 files
    2. Build HNSW and brute-force indices
    3. Compute ground truth using brute-force search
    4. Measure recall before and after deletion/update operations
    5. Run multiple iterations of deletion/update tests
    """

    def __init__(self,
                 data_path: str,
                 num_elements: int = 100_000,
                 num_queries: int = 1_000,
                 k: int = 10,
                 M: int = 16,
                 ef_construction: int = 200,
                 ef_search: int = 200,
                 seed: int = 123):
        """
        Initialize the deletion/update tester.

        Args:
            data_path: Path to the h5 dataset file
            num_elements: Number of elements to use from the dataset
            num_queries: Number of query vectors for testing
            k: Number of nearest neighbors to retrieve
            M: HNSW M parameter (max connections per node)
            ef_construction: HNSW ef_construction parameter
            ef_search: HNSW ef parameter for search
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.num_elements = num_elements
        self.num_queries = num_queries
        self.k = k
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed

        # Set random seed
        np.random.seed(seed)

        # Initialize data containers
        self.data = None
        self.queries = None
        self.ground_truth = None
        self.hnsw_index = None
        self.bf_index = None

        # Results storage
        self.test_results = []

    def load_sift_data(self, force_reload: bool = False) -> np.ndarray:
        """
        Load SIFT dataset from h5 file.

        Args:
            force_reload: If True, reload data even if already loaded

        Returns:
            Loaded data array
        """
        if self.data is not None and not force_reload:
            print("Data already loaded")
            return self.data

        print(f"Loading SIFT dataset from {self.data_path}...")
        with h5py.File(self.data_path, 'r') as f:
            # Load training vectors
            full_data = f['train_vectors'][:]
            print(f"Full dataset shape: {full_data.shape}")

            # Sample the required number of elements
            if len(full_data) > self.num_elements:
                indices = np.random.choice(
                    len(full_data), self.num_elements, replace=False)
                self.data = full_data[indices].astype(np.float32)
            else:
                self.data = full_data[:self.num_elements].astype(np.float32)

            print(f"Loaded data shape: {self.data.shape}")

            # Generate query vectors (different from training data)
            remaining_indices = np.setdiff1d(np.arange(len(full_data)),
                                             indices if len(full_data) > self.num_elements else np.arange(self.num_elements))

            if len(remaining_indices) >= self.num_queries:
                query_indices = np.random.choice(
                    remaining_indices, self.num_queries, replace=False)
                self.queries = full_data[query_indices].astype(np.float32)
            else:
                # If not enough remaining data, use random vectors
                print(
                    "Warning: Not enough data for separate queries, generating random queries")
                self.queries = np.random.rand(
                    self.num_queries, self.data.shape[1]).astype(np.float32)

            print(f"Query data shape: {self.queries.shape}")

        return self.data

    def build_indices(self, allow_replace_deleted: bool = True) -> Tuple[hnswlib.Index, hnswlib.BFIndex]:
        """
        Build HNSW and brute-force indices.

        Args:
            allow_replace_deleted: Whether to allow replacing deleted elements in HNSW

        Returns:
            Tuple of (hnsw_index, bf_index)
        """
        if self.data is None:
            self.load_sift_data()

        dim = self.data.shape[1]

        print("Building HNSW index...")
        self.hnsw_index = hnswlib.Index(space='l2', dim=dim)
        self.hnsw_index.init_index(
            max_elements=self.num_elements * 2,  # Extra space for replacements
            ef_construction=self.ef_construction,
            M=self.M,
            allow_replace_deleted=allow_replace_deleted
        )
        self.hnsw_index.set_ef(self.ef_search)
        self.hnsw_index.set_num_threads(4)

        # Add data to HNSW index
        labels = np.arange(self.num_elements)
        self.hnsw_index.add_items(self.data, labels)

        print("Building brute-force index for ground truth...")
        self.bf_index = hnswlib.BFIndex(space='l2', dim=dim)
        self.bf_index.init_index(max_elements=self.num_elements * 2)
        self.bf_index.add_items(self.data, labels)

        print("Indices built successfully")
        return self.hnsw_index, self.bf_index

    def compute_ground_truth(self) -> np.ndarray:
        """
        Compute ground truth using brute-force search.

        Returns:
            Ground truth labels array of shape (num_queries, k)
        """
        if self.bf_index is None:
            self.build_indices()

        print("Computing ground truth...")
        start_time = time.time()

        labels_gt, _ = self.bf_index.knn_query(self.queries, k=self.k)
        self.ground_truth = labels_gt

        end_time = time.time()
        print(f"Ground truth computed in {end_time - start_time:.2f} seconds")

        return self.ground_truth

    def compute_recall(self, labels_hnsw: np.ndarray, labels_gt: np.ndarray) -> float:
        """
        Compute recall between HNSW results and ground truth.

        Args:
            labels_hnsw: HNSW search results
            labels_gt: Ground truth labels

        Returns:
            Recall score (0.0 to 1.0)
        """
        correct = 0
        total = labels_gt.shape[0] * labels_gt.shape[1]

        for i in range(labels_gt.shape[0]):
            for j in range(labels_gt.shape[1]):
                if labels_gt[i, j] in labels_hnsw[i]:
                    correct += 1

        return correct / total

    def get_recall(self) -> float:
        """
        Get current recall of HNSW index against ground truth.

        Returns:
            Current recall score
        """
        if self.ground_truth is None:
            self.compute_ground_truth()

        labels_hnsw, _ = self.hnsw_index.knn_query(self.queries, k=self.k)
        return self.compute_recall(labels_hnsw, self.ground_truth)

    def test_deletion_impact(self,
                             deletion_ratio: float = 0.1,
                             num_iterations: int = 1) -> List[Dict]:
        """
        Test the impact of deletion on recall.

        Args:
            deletion_ratio: Fraction of elements to delete (0.0 to 1.0)
            num_iterations: Number of test iterations to run

        Returns:
            List of test results for each iteration
        """
        if self.hnsw_index is None:
            self.build_indices()

        results = []

        for iteration in range(num_iterations):
            print(
                f"\n=== Deletion Test Iteration {iteration + 1}/{num_iterations} ===")

            # Get baseline recall
            baseline_recall = self.get_recall()
            print(f"Baseline recall: {baseline_recall:.4f}")

            # Select elements to delete
            num_to_delete = int(self.num_elements * deletion_ratio)
            elements_to_delete = np.random.choice(
                self.num_elements, num_to_delete, replace=False)

            print(
                f"Deleting {num_to_delete} elements ({deletion_ratio:.1%} of data)...")

            # Mark elements as deleted
            start_time = time.time()
            for label in elements_to_delete:
                self.hnsw_index.mark_deleted(label)
            deletion_time = time.time() - start_time

            # Also remove from brute force index for fair comparison
            for label in elements_to_delete:
                self.bf_index.delete_vector(label)

            # Recompute ground truth without deleted elements
            self.compute_ground_truth()

            # Get recall after deletion
            after_deletion_recall = self.get_recall()
            print(f"Recall after deletion: {after_deletion_recall:.4f}")

            result = {
                'iteration': iteration + 1,
                'deletion_ratio': deletion_ratio,
                'num_deleted': num_to_delete,
                'baseline_recall': baseline_recall,
                'after_deletion_recall': after_deletion_recall,
                'recall_drop': baseline_recall - after_deletion_recall,
                'deletion_time': deletion_time,
                'deleted_elements': elements_to_delete.copy()
            }

            results.append(result)

            # Restore deleted elements for next iteration
            if iteration < num_iterations - 1:
                print("Restoring deleted elements for next iteration...")
                # Re-add to brute force
                for i, label in enumerate(elements_to_delete):
                    self.bf_index.add_items(
                        self.data[label:label+1], np.array([label]))

                # Re-add to HNSW (this will unmark them as deleted)
                for i, label in enumerate(elements_to_delete):
                    self.hnsw_index.add_items(
                        self.data[label:label+1], np.array([label]))

                # Recompute ground truth with all elements
                self.compute_ground_truth()

        return results

    def test_replacement_impact(self,
                                replacement_ratio: float = 0.1,
                                num_iterations: int = 1) -> List[Dict]:
        """
        Test the impact of element replacement on recall.

        Args:
            replacement_ratio: Fraction of elements to replace (0.0 to 1.0)
            num_iterations: Number of test iterations to run

        Returns:
            List of test results for each iteration
        """
        if self.hnsw_index is None:
            self.build_indices()

        results = []

        for iteration in range(num_iterations):
            print(
                f"\n=== Replacement Test Iteration {iteration + 1}/{num_iterations} ===")

            # Get baseline recall
            baseline_recall = self.get_recall()
            print(f"Baseline recall: {baseline_recall:.4f}")

            # Select elements to replace
            num_to_replace = int(self.num_elements * replacement_ratio)
            elements_to_replace = np.random.choice(
                self.num_elements, num_to_replace, replace=False)

            # Generate new random data for replacement
            dim = self.data.shape[1]
            new_data = np.random.rand(num_to_replace, dim).astype(np.float32)
            new_labels = np.arange(
                self.num_elements, self.num_elements + num_to_replace)

            print(
                f"Replacing {num_to_replace} elements ({replacement_ratio:.1%} of data)...")

            # Mark old elements as deleted
            start_time = time.time()
            for label in elements_to_replace:
                self.hnsw_index.mark_deleted(label)
                self.bf_index.delete_vector(label)

            # Add new elements using replace_deleted=True
            self.hnsw_index.add_items(
                new_data, new_labels, replace_deleted=True)
            self.bf_index.add_items(new_data, new_labels)

            replacement_time = time.time() - start_time

            # Recompute ground truth with new data
            self.compute_ground_truth()

            # Get recall after replacement
            after_replacement_recall = self.get_recall()
            print(f"Recall after replacement: {after_replacement_recall:.4f}")

            result = {
                'iteration': iteration + 1,
                'replacement_ratio': replacement_ratio,
                'num_replaced': num_to_replace,
                'baseline_recall': baseline_recall,
                'after_replacement_recall': after_replacement_recall,
                'recall_change': after_replacement_recall - baseline_recall,
                'replacement_time': replacement_time,
                'replaced_elements': elements_to_replace.copy(),
                'new_labels': new_labels.copy()
            }

            results.append(result)

            # Restore original data for next iteration
            if iteration < num_iterations - 1:
                print("Restoring original data for next iteration...")
                # Remove new elements
                for label in new_labels:
                    self.hnsw_index.mark_deleted(label)
                    self.bf_index.delete_vector(label)

                # Re-add original elements
                original_data = self.data[elements_to_replace]
                self.hnsw_index.add_items(
                    original_data, elements_to_replace, replace_deleted=True)
                self.bf_index.add_items(original_data, elements_to_replace)

                # Recompute ground truth with original data
                self.compute_ground_truth()

        return results

    def run_comprehensive_test(self,
                               deletion_ratios: List[float] = [
                                   0.05, 0.1, 0.2, 0.3],
                               replacement_ratios: List[float] = [
                                   0.05, 0.1, 0.2, 0.3],
                               num_iterations: int = 3) -> Dict:
        """
        Run comprehensive deletion and replacement tests.

        Args:
            deletion_ratios: List of deletion ratios to test
            replacement_ratios: List of replacement ratios to test
            num_iterations: Number of iterations per test

        Returns:
            Dictionary containing all test results
        """
        print("=== Starting Comprehensive Deletion/Update Tests ===")

        # Ensure indices are built
        if self.hnsw_index is None:
            self.build_indices()

        all_results = {
            'deletion_tests': {},
            'replacement_tests': {},
            'test_parameters': {
                'num_elements': self.num_elements,
                'num_queries': self.num_queries,
                'k': self.k,
                'M': self.M,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search,
                'seed': self.seed
            }
        }

        # Run deletion tests
        print("\n" + "="*50)
        print("DELETION TESTS")
        print("="*50)

        for ratio in deletion_ratios:
            print(f"\nTesting deletion ratio: {ratio:.1%}")
            results = self.test_deletion_impact(ratio, num_iterations)
            all_results['deletion_tests'][ratio] = results

        # Rebuild indices for replacement tests
        print("\nRebuilding indices for replacement tests...")
        self.build_indices()

        # Run replacement tests
        print("\n" + "="*50)
        print("REPLACEMENT TESTS")
        print("="*50)

        for ratio in replacement_ratios:
            print(f"\nTesting replacement ratio: {ratio:.1%}")
            results = self.test_replacement_impact(ratio, num_iterations)
            all_results['replacement_tests'][ratio] = results

        # Store results for later analysis
        self.test_results = all_results

        return all_results

    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot test results showing recall changes.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.test_results:
            print("No test results to plot. Run tests first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot deletion results
        if self.test_results['deletion_tests']:
            deletion_ratios = []
            avg_recall_drops = []
            std_recall_drops = []

            for ratio, results in self.test_results['deletion_tests'].items():
                recall_drops = [r['recall_drop'] for r in results]
                deletion_ratios.append(ratio)
                avg_recall_drops.append(np.mean(recall_drops))
                std_recall_drops.append(np.std(recall_drops))

            ax1.errorbar(deletion_ratios, avg_recall_drops, yerr=std_recall_drops,
                         marker='o', capsize=5, capthick=2, linewidth=2)
            ax1.set_xlabel('Deletion Ratio')
            ax1.set_ylabel('Average Recall Drop')
            ax1.set_title('Impact of Deletion on Recall')
            ax1.grid(True, alpha=0.3)

        # Plot replacement results
        if self.test_results['replacement_tests']:
            replacement_ratios = []
            avg_recall_changes = []
            std_recall_changes = []

            for ratio, results in self.test_results['replacement_tests'].items():
                recall_changes = [r['recall_change'] for r in results]
                replacement_ratios.append(ratio)
                avg_recall_changes.append(np.mean(recall_changes))
                std_recall_changes.append(np.std(recall_changes))

            ax2.errorbar(replacement_ratios, avg_recall_changes, yerr=std_recall_changes,
                         marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
            ax2.set_xlabel('Replacement Ratio')
            ax2.set_ylabel('Average Recall Change')
            ax2.set_title('Impact of Replacement on Recall')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print a summary of test results."""
        if not self.test_results:
            print("No test results available. Run tests first.")
            return

        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)

        # Deletion test summary
        if self.test_results['deletion_tests']:
            print("\nDELETION TESTS:")
            for ratio, results in self.test_results['deletion_tests'].items():
                avg_drop = np.mean([r['recall_drop'] for r in results])
                std_drop = np.std([r['recall_drop'] for r in results])
                avg_time = np.mean([r['deletion_time'] for r in results])

                print(f"  {ratio:.1%} deletion: "
                      f"recall drop = {avg_drop:.4f} ± {std_drop:.4f}, "
                      f"avg time = {avg_time:.3f}s")

        # Replacement test summary
        if self.test_results['replacement_tests']:
            print("\nREPLACEMENT TESTS:")
            for ratio, results in self.test_results['replacement_tests'].items():
                avg_change = np.mean([r['recall_change'] for r in results])
                std_change = np.std([r['recall_change'] for r in results])
                avg_time = np.mean([r['replacement_time'] for r in results])

                print(f"  {ratio:.1%} replacement: "
                      f"recall change = {avg_change:.4f} ± {std_change:.4f}, "
                      f"avg time = {avg_time:.3f}s")


# Example usage
if __name__ == "__main__":
    # Example with SIFT dataset
    tester = DeletionUpdateTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        num_elements=50_000,
        num_queries=1_000,
        k=10,
        M=16,
        ef_construction=200,
        ef_search=200,
        seed=123
    )

    # Load data and build indices
    tester.load_sift_data()
    tester.build_indices()

    # Run comprehensive tests
    results = tester.run_comprehensive_test(
        deletion_ratios=[0.05, 0.1, 0.2],
        replacement_ratios=[0.05, 0.1, 0.2],
        num_iterations=3
    )

    # Print summary and plot results
    tester.print_summary()
    tester.plot_results("deletion_update_results.png")
