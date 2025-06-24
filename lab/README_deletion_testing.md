# HNSW Deletion and Update Testing Framework

This framework provides comprehensive testing capabilities for deletion and update operations on HNSW indices using real datasets like SIFT.

## Overview

The `DeletionUpdateTester` class allows you to:

1. **Load real datasets** (SIFT, YFCC, etc.) from h5 files
2. **Build HNSW and brute-force indices** for comparison
3. **Compute ground truth** using brute-force search
4. **Test deletion impact** on recall performance
5. **Test element replacement** and its effect on recall
6. **Run comprehensive tests** with multiple parameters
7. **Visualize results** with plots and detailed summaries

## Files

- `test_deletion_update.py` - Main class implementing the testing framework
- `example_deletion_test.py` - Example usage scripts with different test scenarios
- `README_deletion_testing.md` - This documentation file

## Quick Start

### Basic Usage

```python
from test_deletion_update import DeletionUpdateTester

# Initialize tester
tester = DeletionUpdateTester(
    data_path='/path/to/sift_dataset.h5',
    num_elements=50_000,     # Number of vectors to use
    num_queries=1_000,       # Number of query vectors
    k=10,                    # Top-k nearest neighbors
    M=16,                    # HNSW M parameter
    ef_construction=200,     # HNSW ef_construction
    ef_search=200,           # HNSW ef for search
    seed=123                 # Random seed
)

# Load data and build indices
tester.load_sift_data()
tester.build_indices()

# Test deletion impact (delete 10% of elements)
deletion_results = tester.test_deletion_impact(deletion_ratio=0.1, num_iterations=3)

# Test replacement impact (replace 10% of elements)
replacement_results = tester.test_replacement_impact(replacement_ratio=0.1, num_iterations=3)
```

### Comprehensive Testing

```python
# Run comprehensive tests with multiple ratios
results = tester.run_comprehensive_test(
    deletion_ratios=[0.05, 0.1, 0.2, 0.3],
    replacement_ratios=[0.05, 0.1, 0.2, 0.3],
    num_iterations=3
)

# Print summary and create plots
tester.print_summary()
tester.plot_results("results.png")
```

## Class Methods

### Data Loading and Index Building

#### `load_sift_data(force_reload=False)`
- Loads SIFT dataset from h5 file
- Samples specified number of elements
- Generates separate query vectors
- Returns: numpy array of loaded data

#### `build_indices(allow_replace_deleted=True)`
- Builds both HNSW and brute-force indices
- Configures HNSW with specified parameters
- Returns: tuple of (hnsw_index, bf_index)

### Ground Truth and Recall Computation

#### `compute_ground_truth()`
- Computes ground truth using brute-force search
- Returns: ground truth labels array

#### `get_recall()`
- Computes current recall of HNSW vs ground truth
- Returns: recall score (0.0 to 1.0)

#### `compute_recall(labels_hnsw, labels_gt)`
- Computes recall between HNSW results and ground truth
- Returns: recall score

### Testing Methods

#### `test_deletion_impact(deletion_ratio=0.1, num_iterations=1)`
- Tests impact of element deletion on recall
- **Parameters:**
  - `deletion_ratio`: Fraction of elements to delete (0.0 to 1.0)
  - `num_iterations`: Number of test iterations
- **Returns:** List of result dictionaries with:
  - `baseline_recall`: Recall before deletion
  - `after_deletion_recall`: Recall after deletion
  - `recall_drop`: Difference in recall
  - `deletion_time`: Time taken for deletion
  - `deleted_elements`: Array of deleted element IDs

#### `test_replacement_impact(replacement_ratio=0.1, num_iterations=1)`
- Tests impact of element replacement on recall
- **Parameters:**
  - `replacement_ratio`: Fraction of elements to replace (0.0 to 1.0)
  - `num_iterations`: Number of test iterations
- **Returns:** List of result dictionaries with:
  - `baseline_recall`: Recall before replacement
  - `after_replacement_recall`: Recall after replacement
  - `recall_change`: Change in recall (can be positive or negative)
  - `replacement_time`: Time taken for replacement
  - `replaced_elements`: Array of replaced element IDs
  - `new_labels`: Array of new element labels

#### `run_comprehensive_test(deletion_ratios, replacement_ratios, num_iterations=3)`
- Runs comprehensive tests with multiple ratios
- **Parameters:**
  - `deletion_ratios`: List of deletion ratios to test
  - `replacement_ratios`: List of replacement ratios to test
  - `num_iterations`: Number of iterations per test
- **Returns:** Dictionary with all test results

### Visualization and Analysis

#### `plot_results(save_path=None)`
- Creates plots showing recall changes
- Shows error bars for multiple iterations
- Optionally saves plot to file

#### `print_summary()`
- Prints comprehensive summary of all test results
- Shows averages and standard deviations
- Includes timing information

## Example Test Scenarios

### 1. Quick Demo Test
```bash
cd lab
python example_deletion_test.py
```
This runs a quick demonstration with smaller parameters.

### 2. Parameter Sensitivity Analysis
Test how different HNSW parameters affect deletion impact:

```python
# Test different M and ef values
m_values = [8, 16, 32]
ef_values = [50, 100, 200]

for M in m_values:
    for ef in ef_values:
        tester = DeletionUpdateTester(M=M, ef_construction=ef, ef_search=ef)
        # ... run tests
```

### 3. High Deletion Rate Test
Test extreme deletion scenarios:

```python
# Test deleting 50% of elements
results = tester.test_deletion_impact(deletion_ratio=0.5, num_iterations=1)
```

### 4. Replacement vs Deletion Comparison
Compare replacement strategy vs simple deletion:

```python
# Test deletion
deletion_results = tester.test_deletion_impact(deletion_ratio=0.2)

# Test replacement  
replacement_results = tester.test_replacement_impact(replacement_ratio=0.2)

# Compare recall changes
```

## Understanding Results

### Deletion Tests
- **Baseline recall**: Recall before any deletions
- **After deletion recall**: Recall after marking elements as deleted
- **Recall drop**: How much recall decreased (always positive)
- **Deletion time**: Time to mark elements as deleted

### Replacement Tests
- **Baseline recall**: Recall before any replacements
- **After replacement recall**: Recall after replacing elements
- **Recall change**: Change in recall (can be positive or negative)
- **Replacement time**: Time to delete old and add new elements

### Key Insights to Look For

1. **Recall degradation**: How much does recall drop with different deletion ratios?
2. **Recovery with replacement**: Does replacing deleted elements help maintain recall?
3. **Parameter sensitivity**: How do different HNSW parameters affect robustness to deletions?
4. **Time complexity**: How does deletion/replacement time scale with the number of operations?

## Configuration Options

### Dataset Parameters
- `data_path`: Path to h5 dataset file
- `num_elements`: Number of vectors to use from dataset
- `num_queries`: Number of query vectors for testing

### HNSW Parameters
- `M`: Maximum number of connections per node
- `ef_construction`: Search parameter during index construction
- `ef_search`: Search parameter during queries

### Test Parameters
- `k`: Number of nearest neighbors to retrieve
- `seed`: Random seed for reproducibility

## Performance Tips

1. **Start small**: Use smaller `num_elements` for initial testing
2. **Adjust ef parameters**: Lower ef values for faster testing, higher for better accuracy
3. **Use appropriate M**: Higher M for better recall but more memory usage
4. **Multiple iterations**: Run 3-5 iterations for statistical significance

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `num_elements` or `num_queries`
2. **Slow performance**: Lower `ef_construction` and `ef_search` values
3. **Low baseline recall**: Increase `ef_search` or `M` parameters
4. **File not found**: Check `data_path` and ensure h5 file exists

### Expected Results

- **Baseline recall**: Should be > 0.9 for good HNSW parameters
- **Deletion impact**: Recall typically drops by 5-15% for 10-20% deletions
- **Replacement benefit**: Should recover most or all recall compared to deletion alone

## Advanced Usage

### Custom Distance Functions
The framework uses L2 distance by default, but you can modify the space parameter:

```python
# In build_indices method, change:
self.hnsw_index = hnswlib.Index(space='cosine', dim=dim)  # or 'ip' for inner product
```

### Different Datasets
To use datasets other than SIFT, modify the `load_sift_data` method or create a new loading method:

```python
def load_custom_data(self):
    # Load your custom dataset
    # Ensure data is float32 and queries are separate
    pass
```

### Custom Metrics
Add custom metrics to the result dictionaries:

```python
# In test methods, add custom measurements
result['custom_metric'] = compute_custom_metric()
```

## Citation

If you use this testing framework in your research, please cite the original HNSW paper and this repository. 