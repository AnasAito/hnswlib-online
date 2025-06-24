import hnswlib
import numpy as np
import h5py
import time
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt


class UpdateOperationTester:
    """
    A class to test update operations on HNSW indices using real datasets.

    Update operation: mark elements as deleted, then re-add the same elements
    to replace the deleted ones. This tests the replace_deleted functionality.
    """

    def __init__(self,
                 data_path: str,
                 num_elements: int = 100_000,
                 num_queries: int = 1_000,
                 k: int = 10,
                 M: int = 16,
                 ef_construction: int = 200,
                 ef_search: int = 200,
                 num_threads: int = 30,
                 seed: int = 123):
        """
        Initialize the update operation tester.

        Args:
            data_path: Path to the h5 dataset file
            num_elements: Number of elements to use from the dataset
            num_queries: Number of query vectors for testing (sampled from training data)
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
        self.num_threads = num_threads
        self.seed = seed

        # Set random seed
        # np.random.seed(seed)

        # Initialize data containers
        self.data = None
        self.queries = None
        self.query_indices = None  # Track which data points are used as queries
        self.ground_truth = None
        self.hnsw_index = None
        self.bf_index = None

        # Track currently deleted elements
        self.currently_deleted: Set[int] = set()

    def load_data(self) -> np.ndarray:
        """
        Load dataset from h5 file and sample queries from training data.
        Ensures queries are always separate from training data.

        Returns:
            Loaded data array
        """
        print(f"Loading dataset from {self.data_path}...")
        with h5py.File(self.data_path, 'r') as f:
            # Load training vectors
            full_data = f['train_vectors'][:]
            print(f"Full dataset shape: {full_data.shape}")

            # First, sample query indices from the full dataset
            total_needed = self.num_elements + self.num_queries

            if len(full_data) < total_needed:
                raise ValueError(
                    f"Dataset too small: need {total_needed} elements but only have {len(full_data)}")

            # Sample indices for queries first
            all_indices = np.arange(len(full_data))
            query_indices_in_full = np.random.choice(
                all_indices, self.num_queries, replace=False)

            # Remove query indices from available indices for training data
            remaining_indices = np.setdiff1d(
                all_indices, query_indices_in_full)

            # Sample training data indices from remaining indices
            if self.num_elements <= len(remaining_indices):
                training_indices_in_full = np.random.choice(
                    remaining_indices, self.num_elements, replace=False)
            else:
                raise ValueError(
                    f"Not enough remaining data for training: need {self.num_elements} but only have {len(remaining_indices)} after removing queries")

            # Extract training data and queries
            self.data = full_data[training_indices_in_full].astype(np.float32)
            self.queries = full_data[query_indices_in_full].astype(np.float32)

            # Create mapping from training data labels (0 to num_elements-1) to original indices
            self.training_indices_in_full = training_indices_in_full
            self.query_indices_in_full = query_indices_in_full

            # For compatibility, create query_indices that map to training data indices (not used anymore)
            self.query_indices = None  # Queries are now separate

            print(f"Training data shape: {self.data.shape}")
            print(f"Query data shape: {self.queries.shape}")
            print(
                f"Sample query indices in full dataset: {query_indices_in_full[:10]}...")
            print(
                f"Sample training indices in full dataset: {training_indices_in_full[:10]}...")
            print("Queries and training data are completely separate")

        return self.data

    def build_indices(self) -> Tuple[hnswlib.Index, hnswlib.BFIndex]:
        """
        Build HNSW and brute-force indices.

        Returns:
            Tuple of (hnsw_index, bf_index)
        """
        if self.data is None:
            self.load_data()

        dim = self.data.shape[1]

        print("Building HNSW index...")
        self.hnsw_index = hnswlib.Index(space='l2', dim=dim)
        self.hnsw_index.init_index(
            max_elements=self.num_elements * 2,  # Extra space for replacements
            ef_construction=self.ef_construction,
            M=self.M,
            allow_replace_deleted=True  # Enable replacement of deleted elements
        )
        self.hnsw_index.set_ef(self.ef_search)
        self.hnsw_index.set_num_threads(self.num_threads)

        # Add data to HNSW index
        labels = np.arange(self.num_elements)
        self.hnsw_index.add_items(self.data, labels)

        print("Building brute-force index for ground truth...")
        self.bf_index = hnswlib.BFIndex(space='l2', dim=dim)
        self.bf_index.init_index(max_elements=self.num_elements)
        self.bf_index.add_items(self.data, labels)

        print("Indices built successfully")

        # Reset deleted elements tracking
        self.currently_deleted.clear()

        return self.hnsw_index, self.bf_index

    def compute_ground_truth(self) -> np.ndarray:
        """
        Compute ground truth using brute-force search.
        Only needs to be computed once since we don't change the actual data.

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

    def compute_recall(self, labels_hnsw: np.ndarray) -> float:
        """
        Compute recall between HNSW results and ground truth.

        Args:
            labels_hnsw: HNSW search results

        Returns:
            Recall score (0.0 to 1.0)
        """
        if self.ground_truth is None:
            self.compute_ground_truth()

        correct = 0
        total = self.ground_truth.shape[0] * self.ground_truth.shape[1]

        for i in range(self.ground_truth.shape[0]):
            for j in range(self.ground_truth.shape[1]):
                if self.ground_truth[i, j] in labels_hnsw[i]:
                    correct += 1

        return correct / total

    def compute_indegree_statistics(self) -> Dict:
        """
        Compute in-degree statistics by inferring in-edges from out-edges.
        Returns statistics about the graph connectivity.

        Returns:
            Dictionary with in-degree statistics
        """
        if self.hnsw_index is None:
            return {}

        # Dictionary to count in-degrees for each node
        indegree_count = {}

        # Iterate through all nodes and get their out-edges
        processed_nodes = 0
        nodes_to_check = []
        # get labels of points index
        labels = self.hnsw_index.get_ids_list()
        for label in labels:
            try:
                # Get neighbors (out-edges) for this node
                neighbors = self.hnsw_index.get_neis_list(label)
                if neighbors is not None and len(neighbors) > 0:
                    nodes_to_check.append(label)
                    processed_nodes += 1
                    # For each neighbor, increment its in-degree
                    for neighbor_label in neighbors:
                        if neighbor_label in indegree_count:
                            indegree_count[neighbor_label] += 1
                        else:
                            indegree_count[neighbor_label] = 1
            except:
                # Node might be deleted or not exist, skip it
                continue

        # Filter to only include nodes that actually exist (have out-edges or in-edges > 0)
        # Keep nodes that either have out-edges or are referenced by others
        active_indegrees = []
        zero_indegree_count = 0

        for label in nodes_to_check:
            try:
                indegree = indegree_count.get(label, 0)

                if indegree == 0:
                    zero_indegree_count += 1
                else:
                    active_indegrees.append(indegree)

            except:
                # Node doesn't exist, skip
                pass

        if len(active_indegrees) == 0:
            return {
                'total_active_nodes': 0,
                'zero_indegree_ratio': 0.0,
                'mean_indegree': 0.0,
                'quantiles': {}
            }

        # Compute statistics
        active_indegrees = np.array(active_indegrees)
        zero_indegree_ratio = zero_indegree_count / self.num_elements
        mean_indegree = np.mean(active_indegrees)

        # Compute quantiles
        quantiles = {}
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            quantiles[f'q{int(q*100)}'] = np.quantile(active_indegrees, q)

        return {
            'total_active_nodes': self.num_elements,
            'processed_nodes': processed_nodes,
            'zero_indegree_count': zero_indegree_count,
            'zero_indegree_ratio': zero_indegree_ratio,
            'mean_indegree': mean_indegree,
            'max_indegree': np.max(active_indegrees),
            'min_indegree': np.min(active_indegrees),
            'std_indegree': np.std(active_indegrees),
            'quantiles': quantiles
        }

    def get_current_recall_with_graph_stats(self) -> Tuple[float, Dict]:
        """
        Get current recall and graph statistics.

        Returns:
            Tuple of (recall, graph_statistics)
        """
        # Compute recall
        labels_hnsw, _ = self.hnsw_index.knn_query(self.queries, k=self.k)
        recall = self.compute_recall(labels_hnsw)

        # Compute graph statistics
        graph_stats = self.compute_indegree_statistics()

        return recall, graph_stats

    def safe_mark_deleted(self, label: int) -> bool:
        """
        Safely mark an element as deleted, avoiding double deletion.

        Args:
            label: Element label to delete

        Returns:
            True if successfully deleted, False if already deleted
        """
        if label in self.currently_deleted:
            return False

        try:
            self.hnsw_index.mark_deleted(label)
            self.currently_deleted.add(label)
            return True
        except Exception as e:
            # print(f"Warning: Could not delete label {label}: {e}")
            return False

    def safe_add_items(self, data: np.ndarray, labels: np.ndarray, replace_deleted: bool = True) -> int:
        """
        Safely add items to the index.

        Args:
            data: Data to add
            labels: Labels for the data
            replace_deleted: Whether to replace deleted elements

        Returns:
            Number of successfully added items
        """
        try:
            self.hnsw_index.add_items(
                data, labels, replace_deleted=replace_deleted)
            # Remove from deleted set since they're now active again
            for label in labels:
                self.currently_deleted.discard(label)
            return len(labels)
        except Exception as e:
            # print(f"Warning: Could not add items: {e}")
            return 0

    def test_single_update(self, update_ratio: float = 0.1) -> Dict:
        """
        Test a single update operation: mark elements as deleted, then re-add them.

        Args:
            update_ratio: Fraction of elements to update (0.0 to 1.0)

        Returns:
            Dictionary with test results
        """
        if self.hnsw_index is None:
            self.build_indices()

        print(f"\n=== Single Update Test (ratio: {update_ratio:.1%}) ===")

        # Get baseline recall
        baseline_recall = self.get_current_recall_with_graph_stats()[0]
        print(f"Baseline recall: {baseline_recall:.4f}")

        # Get baseline graph statistics
        _, baseline_graph_stats = self.get_current_recall_with_graph_stats()
        self.print_graph_statistics(baseline_graph_stats, "Baseline ")

        # Select elements to update (avoid currently deleted ones)
        available_elements = [i for i in range(
            self.num_elements) if i not in self.currently_deleted]
        num_to_update = min(
            int(self.num_elements * update_ratio), len(available_elements))

        if num_to_update == 0:
            print("No available elements to update!")
            return {}

        elements_to_update = np.random.choice(
            available_elements, num_to_update, replace=False)

        print(
            f"Updating {num_to_update} elements ({num_to_update/self.num_elements:.1%} of data)...")

        # Step 1: Mark elements as deleted
        start_time = time.time()
        successfully_deleted = []
        for label in elements_to_update:
            if self.safe_mark_deleted(label):
                successfully_deleted.append(label)

        # Get recall after deletion
        after_deletion_recall, after_deletion_graph_stats = self.get_current_recall_with_graph_stats()
        deletion_time = time.time() - start_time

        print(f"Successfully deleted {len(successfully_deleted)} elements")
        print(f"Recall after deletion: {after_deletion_recall:.4f}")
        self.print_graph_statistics(
            after_deletion_graph_stats, "After deletion ")

        # Step 2: Re-add the same elements using replace_deleted=True
        if successfully_deleted:
            update_data = self.data[successfully_deleted]
            start_time = time.time()
            added_count = self.safe_add_items(update_data, np.array(
                successfully_deleted), replace_deleted=True)
            update_time = time.time() - start_time
            print(f"Successfully re-added {added_count} elements")
        else:
            update_time = 0
            print("No elements to re-add")

        # Get recall after re-adding
        after_update_recall, after_update_graph_stats = self.get_current_recall_with_graph_stats()

        total_time = deletion_time + update_time

        print(f"Recall after deletion: {after_deletion_recall:.4f}")
        print(f"Recall after update: {after_update_recall:.4f}")
        self.print_graph_statistics(after_update_graph_stats, "After update ")
        print(f"Total update time: {total_time:.3f} seconds")

        result = {
            'update_ratio': update_ratio,
            'num_requested': num_to_update,
            'num_deleted': len(successfully_deleted),
            'baseline_recall': baseline_recall,
            'after_deletion_recall': after_deletion_recall,
            'after_update_recall': after_update_recall,
            'recall_drop_from_deletion': baseline_recall - after_deletion_recall,
            'recall_recovery': after_update_recall - after_deletion_recall,
            'final_recall_difference': after_update_recall - baseline_recall,
            'deletion_time': deletion_time,
            'update_time': update_time,
            'total_time': total_time,
            'updated_elements': np.array(successfully_deleted),
            'after_deletion_graph_stats': after_deletion_graph_stats,
            'after_update_graph_stats': after_update_graph_stats
        }

        return result

    def test_multiple_updates(self,
                              update_ratio: float = 0.1,
                              num_iterations: int = 5) -> List[Dict]:
        """
        Test multiple update operations in sequence.

        Args:
            update_ratio: Fraction of elements to update in each iteration
            num_iterations: Number of update iterations to perform

        Returns:
            List of test results for each iteration
        """
        if self.hnsw_index is None:
            self.build_indices()

        print(
            f"\n=== Multiple Updates Test ({num_iterations} iterations, ratio: {update_ratio:.1%}) ===")

        results = []

        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # Get current recall before this iteration
            current_recall, current_graph_stats = self.get_current_recall_with_graph_stats()
            print(f"Current recall: {current_recall:.4f}")
            print(f"Currently deleted elements: {len(self.currently_deleted)}")
            self.print_graph_statistics(
                current_graph_stats, f"Iteration {iteration + 1} start ")

            # Select elements to update (avoid currently deleted ones)
            available_elements = [i for i in range(
                self.num_elements) if i not in self.currently_deleted]
            num_to_update = min(
                int(self.num_elements * update_ratio), len(available_elements))

            if num_to_update == 0:
                print("No available elements to update!")
                break

            elements_to_update = np.random.choice(
                available_elements, num_to_update, replace=False)

            print(f"Updating {num_to_update} elements...")

            # Step 1: Mark elements as deleted
            start_time = time.time()
            successfully_deleted = []
            for label in elements_to_update:
                if self.safe_mark_deleted(label):
                    successfully_deleted.append(label)

            # Get recall after deletion
            after_deletion_recall, after_deletion_graph_stats = self.get_current_recall_with_graph_stats()
            deletion_time = time.time() - start_time

            # Step 2: Re-add the same elements
            if successfully_deleted:
                update_data = self.data[successfully_deleted]
                start_time = time.time()
                added_count = self.safe_add_items(update_data, np.array(
                    successfully_deleted), replace_deleted=True)
                update_time = time.time() - start_time
            else:
                update_time = 0
                added_count = 0

            # Get recall after re-adding
            after_update_recall, after_update_graph_stats = self.get_current_recall_with_graph_stats()

            total_time = deletion_time + update_time

            print(
                f"  Deleted: {len(successfully_deleted)}, Re-added: {added_count}")
            print(f"  After deletion: {after_deletion_recall:.4f}")
            print(f"  After update: {after_update_recall:.4f}")
            print(f"  Time: {total_time:.3f}s")

            result = {
                'iteration': iteration + 1,
                'update_ratio': update_ratio,
                'num_requested': num_to_update,
                'num_deleted': len(successfully_deleted),
                'before_recall': current_recall,
                'after_deletion_recall': after_deletion_recall,
                'after_update_recall': after_update_recall,
                'recall_drop_from_deletion': current_recall - after_deletion_recall,
                'recall_recovery': after_update_recall - after_deletion_recall,
                'final_recall_difference': after_update_recall - current_recall,
                'deletion_time': deletion_time,
                'update_time': update_time,
                'total_time': total_time,
                'updated_elements': np.array(successfully_deleted),
                'after_deletion_graph_stats': after_deletion_graph_stats,
                'after_update_graph_stats': after_update_graph_stats
            }

            results.append(result)

        # Print summary
        print(f"\n=== Summary of {len(results)} iterations ===")
        if results:
            avg_recall_drop = np.mean(
                [r['recall_drop_from_deletion'] for r in results])
            avg_recall_recovery = np.mean(
                [r['recall_recovery'] for r in results])
            avg_final_diff = np.mean(
                [r['final_recall_difference'] for r in results])
            avg_total_time = np.mean([r['total_time'] for r in results])

            print(f"Average recall drop from deletion: {avg_recall_drop:.4f}")
            print(f"Average recall recovery: {avg_recall_recovery:.4f}")
            print(f"Average final recall difference: {avg_final_diff:.4f}")
            print(f"Average total time per iteration: {avg_total_time:.3f}s")

            # Graph statistics summary
            if 'after_update_graph_stats' in results[0]:
                print(f"\n--- Graph Structure Changes ---")
                zero_indegree_ratios = [
                    r['after_update_graph_stats']['zero_indegree_ratio'] for r in results]
                mean_indegrees = [r['after_update_graph_stats']
                                  ['mean_indegree'] for r in results]

                print(
                    f"Zero in-degree ratio: {np.mean(zero_indegree_ratios):.4f} ± {np.std(zero_indegree_ratios):.4f}")
                print(
                    f"Mean in-degree: {np.mean(mean_indegrees):.2f} ± {np.std(mean_indegrees):.2f}")

                # Show trend
                if len(results) > 1:
                    initial_zero_ratio = results[0]['after_update_graph_stats']['zero_indegree_ratio']
                    final_zero_ratio = results[-1]['after_update_graph_stats']['zero_indegree_ratio']
                    print(
                        f"Zero in-degree ratio change: {initial_zero_ratio:.4f} → {final_zero_ratio:.4f}")

        return results

    def plot_update_results(self, results: List[Dict], save_path: str = None):
        """
        Plot results from multiple update iterations.

        Args:
            results: List of results from test_multiple_updates
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to plot!")
            return

        iterations = [r['iteration'] for r in results]
        before_recalls = [r['before_recall'] for r in results]
        after_deletion_recalls = [r['after_deletion_recall']
                                  for r in results]
        after_update_recalls = [r['after_update_recall'] for r in results]

        # Check if we have graph statistics
        has_graph_stats = 'after_update_graph_stats' in results[0]

        # Set up figure with 3 subplots
        plt.figure(figsize=(18, 6))
        subplot_cols = 3

        # Plot 1: Recall progression only
        plt.subplot(1, subplot_cols, 1)
        plt.plot(iterations[:15], after_update_recalls[:15], '^-',
                 label='After Update', linewidth=2, markersize=6)
        plt.xlabel('Iteration')
        plt.ylabel('Recall')
        plt.title('Recall During Update Operations')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Graph statistics with dual y-axis (both zero in-degree ratio and mean in-degree)
        if has_graph_stats:
            plt.subplot(1, subplot_cols, 2)
            zero_indegree_ratios = [
                r['after_update_graph_stats']['zero_indegree_ratio'] for r in results]
            mean_indegrees = [r['after_update_graph_stats']
                              ['mean_indegree'] for r in results]

            # Dual y-axis plot
            ax1 = plt.gca()
            color = 'tab:red'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Zero In-degree Ratio', color=color)
            line1 = ax1.plot(iterations[:15], zero_indegree_ratios[:15], 'o-', color=color,
                             linewidth=2, markersize=6, label='Zero In-degree Ratio')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)

            # ax2 = ax1.twinx()
            # color = 'tab:blue'
            # ax2.set_ylabel('Mean In-degree', color=color)
            # line2 = ax2.plot(iterations, mean_indegrees, 's-', color=color,
            #                  linewidth=2, markersize=6, label='Mean In-degree')
            # ax2.tick_params(axis='y', labelcolor=color)

            # Add combined legend
            lines = line1
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            plt.title('Graph Structure Evolution')

            # Plot 3: Deltas (differences between consecutive iterations)
            plt.subplot(1, subplot_cols, 3)

            if len(iterations) > 1:
                # Calculate deltas for recall
                delta_recalls = [after_update_recalls[i] - after_update_recalls[i-1]
                                 for i in range(1, len(after_update_recalls))]

                # Calculate deltas for zero in-degree ratio
                delta_zero_indegree = [zero_indegree_ratios[i] - zero_indegree_ratios[i-1]
                                       for i in range(1, len(zero_indegree_ratios))]

                # Iterations for delta plots (start from 2nd iteration)
                delta_iterations = iterations[1:]

                # Dual y-axis for deltas
                ax3 = plt.gca()
                color = 'tab:green'
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Δ Recall', color=color)
                line3 = ax3.plot(delta_iterations, delta_recalls, '^-', color=color,
                                 linewidth=2, markersize=6, label='Δ Recall')
                ax3.tick_params(axis='y', labelcolor=color)
                ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color=color, linestyle='--', alpha=0.5)

                ax4 = ax3.twinx()
                color = 'tab:orange'
                ax4.set_ylabel('Δ Zero In-degree Ratio', color=color)
                line4 = ax4.plot(delta_iterations, delta_zero_indegree, 'o-', color=color,
                                 linewidth=2, markersize=6, label='Δ Zero In-degree Ratio')
                ax4.tick_params(axis='y', labelcolor=color)
                ax4.axhline(y=0, color=color, linestyle='--', alpha=0.5)

                # Add combined legend
                lines = line3 + line4
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='upper left')

                plt.title('Delta Changes Between Iterations')
            else:
                plt.text(0.5, 0.5, 'Need at least 2 iterations\nfor delta calculation',
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Delta Changes Between Iterations')
        else:
            # If no graph stats, just show placeholder for subplots 2 and 3
            plt.subplot(1, subplot_cols, 2)
            plt.text(0.5, 0.5, 'No graph statistics available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Graph Structure Evolution')

            plt.subplot(1, subplot_cols, 3)
            plt.text(0.5, 0.5, 'No graph statistics available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Delta Changes Between Iterations')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def test_replacement_operation(self, replacement_ratio: float = 0.1) -> Dict:
        """
        Test a replacement operation: mark elements as deleted, then add DIFFERENT elements.
        This requires recomputing ground truth since the actual data changes.

        Args:
            replacement_ratio: Fraction of elements to replace (0.0 to 1.0)

        Returns:
            Dictionary with test results
        """
        if self.hnsw_index is None:
            self.build_indices()

        print(
            f"\n=== Replacement Operation Test (ratio: {replacement_ratio:.1%}) ===")

        # Get baseline recall
        baseline_recall = self.get_current_recall_with_graph_stats()[0]
        print(f"Baseline recall: {baseline_recall:.4f}")

        # Select elements to delete (avoid currently deleted ones)
        available_elements = [i for i in range(
            self.num_elements) if i not in self.currently_deleted]
        num_to_replace = min(
            int(self.num_elements * replacement_ratio), len(available_elements))

        if num_to_replace == 0:
            print("No available elements to replace!")
            return {}

        elements_to_delete = np.random.choice(
            available_elements, num_to_replace, replace=False)

        print(
            f"Replacing {num_to_replace} elements ({num_to_replace/self.num_elements:.1%} of data)...")

        # Step 1: Mark elements as deleted
        start_time = time.time()
        successfully_deleted = []
        for label in elements_to_delete:
            if self.safe_mark_deleted(label):
                successfully_deleted.append(label)

        # Also remove from brute force index for ground truth recomputation
        for label in successfully_deleted:
            try:
                self.bf_index.delete_vector(label)
            except:
                pass  # Ignore if already deleted

        deletion_time = time.time() - start_time
        print(f"Successfully deleted {len(successfully_deleted)} elements")

        # Recompute ground truth after deletion
        print("Recomputing ground truth after deletion...")
        self.compute_ground_truth()

        # Get recall after deletion
        after_deletion_recall = self.get_current_recall_with_graph_stats()[0]

        # Step 2: Generate new random data and add it
        if successfully_deleted:
            # Generate completely new random data
            dim = self.data.shape[1]
            new_data = np.random.rand(
                len(successfully_deleted), dim).astype(np.float32)

            # Use new labels starting from a high number to avoid conflicts
            new_labels = np.arange(
                self.num_elements * 10, self.num_elements * 10 + len(successfully_deleted))

            start_time = time.time()

            # Add to HNSW with replace_deleted=True
            added_count = self.safe_add_items(
                new_data, new_labels, replace_deleted=True)

            # Add to brute force index as well
            try:
                self.bf_index.add_items(new_data, new_labels)
            except:
                pass  # Handle any errors gracefully

            replacement_time = time.time() - start_time
            print(
                f"Successfully added {added_count} new elements with labels {new_labels[:5]}...")

            # Recompute ground truth after adding new elements
            print("Recomputing ground truth after replacement...")
            self.compute_ground_truth()
        else:
            replacement_time = 0
            new_labels = np.array([])
            print("No elements to replace")

        # Get recall after replacement
        after_replacement_recall = self.get_current_recall_with_graph_stats()[
            0]

        total_time = deletion_time + replacement_time

        print(f"Recall after deletion: {after_deletion_recall:.4f}")
        print(f"Recall after replacement: {after_replacement_recall:.4f}")
        print(f"Total replacement time: {total_time:.3f} seconds")

        result = {
            'replacement_ratio': replacement_ratio,
            'num_requested': num_to_replace,
            'num_deleted': len(successfully_deleted),
            'num_added': len(new_labels) if len(successfully_deleted) > 0 else 0,
            'baseline_recall': baseline_recall,
            'after_deletion_recall': after_deletion_recall,
            'after_replacement_recall': after_replacement_recall,
            'recall_drop_from_deletion': baseline_recall - after_deletion_recall,
            'recall_change_from_replacement': after_replacement_recall - after_deletion_recall,
            'final_recall_difference': after_replacement_recall - baseline_recall,
            'deletion_time': deletion_time,
            'replacement_time': replacement_time,
            'total_time': total_time,
            'deleted_elements': np.array(successfully_deleted),
            'new_labels': new_labels
        }

        return result

    def test_multiple_replacements(self,
                                   replacement_ratio: float = 0.1,
                                   num_iterations: int = 5) -> List[Dict]:
        """
        Test multiple replacement operations in sequence.
        Each iteration deletes some elements and adds different elements.

        Args:
            replacement_ratio: Fraction of elements to replace in each iteration
            num_iterations: Number of replacement iterations to perform

        Returns:
            List of test results for each iteration
        """
        if self.hnsw_index is None:
            self.build_indices()

        print(
            f"\n=== Multiple Replacements Test ({num_iterations} iterations, ratio: {replacement_ratio:.1%}) ===")

        results = []

        for iteration in range(num_iterations):
            print(
                f"\n--- Replacement Iteration {iteration + 1}/{num_iterations} ---")

            # Get current recall before this iteration
            current_recall, current_graph_stats = self.get_current_recall_with_graph_stats()
            print(f"Current recall: {current_recall:.4f}")
            print(f"Currently deleted elements: {len(self.currently_deleted)}")

            # Select elements to delete (avoid currently deleted ones)
            available_elements = [i for i in range(
                self.num_elements) if i not in self.currently_deleted]
            num_to_replace = min(
                int(self.num_elements * replacement_ratio), len(available_elements))

            if num_to_replace == 0:
                print("No available elements to replace!")
                break

            elements_to_delete = np.random.choice(
                available_elements, num_to_replace, replace=False)

            print(f"Replacing {num_to_replace} elements...")

            # Step 1: Mark elements as deleted
            start_time = time.time()
            successfully_deleted = []
            for label in elements_to_delete:
                if self.safe_mark_deleted(label):
                    successfully_deleted.append(label)

            # Also remove from brute force index
            for label in successfully_deleted:
                try:
                    self.bf_index.delete_vector(label)
                except:
                    pass

            deletion_time = time.time() - start_time

            # Recompute ground truth after deletion
            self.compute_ground_truth()
            after_deletion_recall, after_deletion_graph_stats = self.get_current_recall_with_graph_stats()

            # Step 2: Add new different elements
            if successfully_deleted:
                # Generate new random data
                dim = self.data.shape[1]
                new_data = np.random.rand(
                    len(successfully_deleted), dim).astype(np.float32)

                # Use new labels to avoid conflicts
                base_label = self.num_elements * 10 + iteration * 100000
                new_labels = np.arange(
                    base_label, base_label + len(successfully_deleted))

                start_time = time.time()

                # Add to both indices
                added_count = self.safe_add_items(
                    new_data, new_labels, replace_deleted=True)
                try:
                    self.bf_index.add_items(new_data, new_labels)
                except:
                    pass

                replacement_time = time.time() - start_time

                # Recompute ground truth after replacement
                self.compute_ground_truth()
            else:
                replacement_time = 0
                added_count = 0
                new_labels = np.array([])

            # Get recall after replacement
            after_replacement_recall, after_replacement_graph_stats = self.get_current_recall_with_graph_stats()

            total_time = deletion_time + replacement_time

            print(
                f"  Deleted: {len(successfully_deleted)}, Added: {added_count}")
            print(f"  After deletion: {after_deletion_recall:.4f}")
            print(f"  After replacement: {after_replacement_recall:.4f}")
            print(f"  Time: {total_time:.3f}s")

            result = {
                'iteration': iteration + 1,
                'replacement_ratio': replacement_ratio,
                'num_requested': num_to_replace,
                'num_deleted': len(successfully_deleted),
                'num_added': added_count,
                'before_recall': current_recall,
                'after_deletion_recall': after_deletion_recall,
                'after_replacement_recall': after_replacement_recall,
                'recall_drop_from_deletion': current_recall - after_deletion_recall,
                'recall_change_from_replacement': after_replacement_recall - after_deletion_recall,
                'final_recall_difference': after_replacement_recall - current_recall,
                'deletion_time': deletion_time,
                'replacement_time': replacement_time,
                'total_time': total_time,
                'deleted_elements': np.array(successfully_deleted),
                'new_labels': new_labels,
                'after_deletion_graph_stats': after_deletion_graph_stats,
                'after_replacement_graph_stats': after_replacement_graph_stats
            }

            results.append(result)

        # Print summary
        print(f"\n=== Summary of {len(results)} replacement iterations ===")
        if results:
            avg_recall_drop = np.mean(
                [r['recall_drop_from_deletion'] for r in results])
            avg_recall_change = np.mean(
                [r['recall_change_from_replacement'] for r in results])
            avg_final_diff = np.mean(
                [r['final_recall_difference'] for r in results])
            avg_total_time = np.mean([r['total_time'] for r in results])

            print(f"Average recall drop from deletion: {avg_recall_drop:.4f}")
            print(
                f"Average recall change from replacement: {avg_recall_change:.4f}")
            print(f"Average final recall difference: {avg_final_diff:.4f}")
            print(f"Average total time per iteration: {avg_total_time:.3f}s")

        return results

    def plot_replacement_results(self, results: List[Dict], save_path: str = None):
        """
        Plot results from multiple replacement iterations.

        Args:
            results: List of results from test_multiple_replacements
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to plot!")
            return

        iterations = [r['iteration'] for r in results]
        before_recalls = [r['before_recall'] for r in results]
        after_deletion_recalls = [r['after_deletion_recall'] for r in results]
        after_replacement_recalls = [
            r['after_replacement_recall'] for r in results]

        plt.figure(figsize=(15, 6))

        # Plot recall progression
        plt.subplot(1, 3, 1)
        plt.plot(iterations, before_recalls, 'o-',
                 label='Before Operation', linewidth=2, markersize=6)
        plt.plot(iterations, after_deletion_recalls, 's-',
                 label='After Deletion', linewidth=2, markersize=6)
        plt.plot(iterations, after_replacement_recalls, '^-',
                 label='After Replacement', linewidth=2, markersize=6)
        plt.xlabel('Iteration')
        plt.ylabel('Recall')
        plt.title('Recall During Replacement Operations')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot timing
        plt.subplot(1, 3, 2)
        deletion_times = [r['deletion_time'] for r in results]
        replacement_times = [r['replacement_time'] for r in results]

        plt.bar([i - 0.2 for i in iterations], deletion_times,
                width=0.4, label='Deletion Time', alpha=0.7)
        plt.bar([i + 0.2 for i in iterations], replacement_times,
                width=0.4, label='Replacement Time', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.title('Replacement Operation Timing')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot recall changes
        plt.subplot(1, 3, 3)
        recall_drops = [r['recall_drop_from_deletion'] for r in results]
        recall_changes = [r['recall_change_from_replacement'] for r in results]
        final_diffs = [r['final_recall_difference'] for r in results]

        plt.plot(iterations, recall_drops, 'v-',
                 label='Drop from Deletion', linewidth=2, markersize=6)
        plt.plot(iterations, recall_changes, '^-',
                 label='Change from Replacement', linewidth=2, markersize=6)
        plt.plot(iterations, final_diffs, 'o-',
                 label='Net Change', linewidth=2, markersize=6)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Recall Change')
        plt.title('Recall Change Components')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def get_current_recall(self) -> float:
        """
        Get current recall of HNSW index against ground truth.

        Returns:
            Current recall score
        """
        labels_hnsw, _ = self.hnsw_index.knn_query(self.queries, k=self.k)
        return self.compute_recall(labels_hnsw)

    def print_graph_statistics(self, graph_stats: Dict, prefix: str = ""):
        """
        Print graph statistics in a readable format.

        Args:
            graph_stats: Dictionary with graph statistics
            prefix: Prefix for the print statements
        """
        if not graph_stats:
            print(f"{prefix}No graph statistics available")
            return

        print(f"{prefix}Graph Statistics:")
        print(f"{prefix}  Active nodes: {graph_stats['total_active_nodes']}")
        print(
            f"{prefix}  Zero in-degree: {graph_stats['zero_indegree_count']} ({graph_stats['zero_indegree_ratio']:.3f})")
        print(f"{prefix}  Mean in-degree: {graph_stats['mean_indegree']:.2f}")
        print(
            f"{prefix}  In-degree range: [{graph_stats['min_indegree']}, {graph_stats['max_indegree']}]")

        if 'quantiles' in graph_stats:
            q_str = ", ".join(
                [f"q{k[1:]}: {v:.1f}" for k, v in graph_stats['quantiles'].items()])
            print(f"{prefix}  Quantiles: {q_str}")


# Example usage
if __name__ == "__main__":
    import json
    import pickle
    use_cache = True
    update_ratio = 0.2
    # Initialize tester
    tester = UpdateOperationTester(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        # data_path='/home/anas.aitaomar/yfcc/yfcc_10m_old_dist.h5',
        num_elements=900_000,
        num_queries=10000,
        k=10,
        M=8,
        ef_construction=100,
        ef_search=100,
        num_threads=30,
        seed=123
    )
    if not use_cache:
        # Load data and build indices
        tester.load_data()
        tester.build_indices()

        # Test single update operation
        single_result = tester.test_single_update(
            update_ratio=update_ratio)

        # Test multiple update operations
        multiple_results = tester.test_multiple_updates(
            update_ratio=update_ratio, num_iterations=15)
        # save results to pickle
        with open(f"update_operations_results_M{tester.M}_efc{tester.ef_construction}_efs{tester.ef_search}_num_elements{tester.num_elements}.pkl", "wb") as f:
            pickle.dump(multiple_results, f)
    else:
        with open(f"update_operations_results_M{tester.M}_efc{tester.ef_construction}_efs{tester.ef_search}_num_elements{tester.num_elements}.pkl", "rb") as f:
            multiple_results = pickle.load(f)

    # Plot results
    tester.plot_update_results(
        multiple_results, f"update_operations_results_M{tester.M}_efc{tester.ef_construction}_efs{tester.ef_search}_num_elements{tester.num_elements}.png")
