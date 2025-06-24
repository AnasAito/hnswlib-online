import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from hnsw_pipeline import HNSWPipeline

out_degree_bound = 32
ef_construction = 100
num_nodes = 2_000_000
use_cached = False  # Set to True to use cached data, False to regenerate

# Create cache signature
cache_signature = f"nodes_{num_nodes}_outdeg_{out_degree_bound}_ef_{ef_construction}"
cache_dir = "degree_cache"
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, f"graph_degrees_{cache_signature}.pkl")

if use_cached and os.path.exists(cache_file):
    print(f"Loading cached graph degree data from {cache_file}")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
        indegrees = cached_data['indegrees']
        outdegrees = cached_data['outdegrees']
    print(f"Loaded graph with {len(indegrees)} nodes")
else:
    print("Building graph and calculating degrees...")
    pipeline = HNSWPipeline(
        # data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        data_path='/home/anas.aitaomar/yfcc/yfcc_10m_old_dist.h5',
        output_dir='pipeline_outputs',
        num_nodes=num_nodes,
        out_degree=out_degree_bound,
        ef_construction=ef_construction,
        seed=None
    )
    pipeline.step1_load_data(force_reload=True)
    pipeline.step2_build_index(force_rebuild=True)
    adj_matrix = pipeline.step3_create_adjacency(force_rebuild=True)

    # For NetworkX ≥ 3.0 use from_scipy_sparse_array; on ≤ 2.8 use from_scipy_sparse_matrix
    graph = nx.from_scipy_sparse_array(
        adj_matrix,
        create_using=nx.DiGraph,          # <- make it directed
        # one edge (i→j); its value becomes weight
        parallel_edges=False,
        edge_attribute="weight"           # store A[i,j] as edge weight
    )

    # Calculate indegrees and outdegrees for all nodes
    indegrees = [d for n, d in graph.in_degree()]
    outdegrees = [d for n, d in graph.out_degree()]

    # Cache the data
    print(f"Caching graph degree data to {cache_file}")
    cache_data = {
        'indegrees': indegrees,
        'outdegrees': outdegrees,
        'signature': cache_signature
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("Data cached successfully")

# Create the plot
plt.figure(figsize=(12, 7))

# Calculate bins with equal width
max_degree = max(max(indegrees), max(outdegrees))
bins = np.arange(0, max_degree + 2) - 0.5  # -0.5 to center bins on integers

# Calculate histograms
indegree_hist, _ = np.histogram(indegrees, bins=bins)
outdegree_hist, _ = np.histogram(outdegrees, bins=bins)

# Plot both distributions
plt.bar(bins[:-1], indegree_hist, width=1,
        alpha=0.5, color='blue', label='Indegree')
plt.bar(bins[:-1], outdegree_hist, width=1,
        alpha=0.5, color='red', label='Outdegree')

plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
plt.grid(True, alpha=0.3)

# Set x-axis to start from 0 and show integer ticks
plt.xlim(0, max_degree + 1)
plt.xticks(np.arange(0, max_degree + 2, 1))

# Add some statistics to the plot
mean_indegree = np.mean(indegrees)
median_indegree = np.median(indegrees)
max_indegree = max(indegrees)
min_indegree = min(indegrees)

mean_outdegree = np.mean(outdegrees)
median_outdegree = np.median(outdegrees)
max_outdegree = max(outdegrees)
min_outdegree = min(outdegrees)

plt.axvline(mean_indegree, color='blue', linestyle='--',
            linewidth=2, label=f'Mean Indegree: {mean_indegree:.2f}')
plt.axvline(mean_outdegree, color='red', linestyle='--',
            linewidth=2, label=f'Mean Outdegree: {mean_outdegree:.2f}')
plt.axvline(out_degree_bound*2, color='green', linestyle='--',
            linewidth=2, label=f'Out-degree bound: {out_degree_bound*2}')
plt.legend()

# Print statistics
print(f"Indegree Statistics:")
print(f"Mean: {mean_indegree:.2f}")
print(f"Median: {median_indegree:.2f}")
print(f"Min: {min_indegree}")
print(f"Max: {max_indegree}")
print(f"Standard deviation: {np.std(indegrees):.2f}")

print(f"\nOutdegree Statistics:")
print(f"Mean: {mean_outdegree:.2f}")
print(f"Median: {median_outdegree:.2f}")
print(f"Min: {min_outdegree}")
print(f"Max: {max_outdegree}")
print(f"Standard deviation: {np.std(outdegrees):.2f}")

plt.tight_layout()
path = os.path.join(
    pipeline.output_dir if 'pipeline' in locals() else 'pipeline_outputs',
    f"degree_distribution_{out_degree_bound}_ef_construction_{ef_construction}_num_nodes_{num_nodes}.png")
plt.savefig(path,
            dpi=300, bbox_inches='tight')
plt.show()

# Analyze indegree distribution across time ranges
print("\n" + "="*50)
print("INDEGREE DISTRIBUTION ACROSS TIME RANGES")
print("="*50)

# Define fixed 250k time ranges
range_size = 500_000
time_ranges = []
start = 1
while start <= num_nodes:
    end = min(start + range_size - 1, num_nodes)
    quarter_num = (start - 1) // range_size + 1
    time_ranges.append(
        (start, end, f"Range {quarter_num} ({start:,}-{end:,})"))
    start = end + 1

print(f"Created {len(time_ranges)} time ranges of {range_size:,} nodes each")

# Create a new figure for time-based analysis
plt.figure(figsize=(14, 8))

colors = ['blue', 'green', 'orange', 'red', 'purple',
          'brown', 'pink', 'gray', 'olive', 'cyan']
alphas = [0.6] * len(time_ranges)

# Calculate indegrees for each time range
time_indegrees = []
for i, (start, end, label) in enumerate(time_ranges):
    # Get nodes in this time range
    nodes_in_range = list(range(start, end + 1))

    # Filter nodes that actually exist in the graph
    existing_nodes = [n for n in nodes_in_range if graph.has_node(n)]

    # Get indegrees for these nodes
    range_indegrees = [graph.in_degree(n) for n in existing_nodes]
    time_indegrees.append(range_indegrees)

    print(f"{label}: {len(existing_nodes)} nodes, mean indegree: {np.mean(range_indegrees):.2f}")

# Find max indegree across all time ranges for consistent binning
max_indegree_time = max(max(indegrees)
                        for indegrees in time_indegrees if indegrees)
bins_time = np.arange(0, max_indegree_time + 2) - 0.5

# Plot the full distribution as reference (thin black line)
full_hist, _ = np.histogram(indegrees, bins=bins_time)
plt.plot(bins_time[:-1], full_hist, color='black', linewidth=2,
         label='Full Distribution (Reference)', alpha=0.8)

# Plot histograms for each time range
for i, (indegrees_range, (_, _, label)) in enumerate(zip(time_indegrees, time_ranges)):
    if indegrees_range:  # Only plot if we have data
        hist, _ = np.histogram(indegrees_range, bins=bins_time)
        color = colors[i % len(colors)]
        plt.bar(bins_time[:-1], hist, width=1, alpha=alphas[i],
                color=color, label=label)

plt.xlabel('Indegree')
plt.ylabel('Frequency')
plt.title('Indegree Distribution Across Time Ranges (250k nodes each)')
plt.grid(True, alpha=0.3)
plt.xlim(0, max_indegree_time + 1)
plt.xticks(np.arange(0, max_indegree_time + 2, 1))
plt.legend()

# Add statistics
for i, (indegrees_range, (_, _, label)) in enumerate(zip(time_indegrees, time_ranges)):
    if indegrees_range:
        mean_val = np.mean(indegrees_range)
        color = colors[i % len(colors)]
        plt.axvline(mean_val, color=color, linestyle='--',
                    linewidth=1.5, alpha=0.8)

# Add full distribution mean as reference
full_mean = np.mean(indegrees)
plt.axvline(full_mean, color='black', linestyle='-',
            linewidth=2, alpha=0.8, label=f'Full Mean: {full_mean:.2f}')

plt.tight_layout()
time_path = os.path.join(
    pipeline.output_dir if 'pipeline' in locals() else 'pipeline_outputs',
    f"indegree_time_distribution_{out_degree_bound}_ef_construction_{ef_construction}_num_nodes_{num_nodes}.png")
plt.savefig(time_path, dpi=300, bbox_inches='tight')
plt.show()

# Analyze outdegree distribution across time ranges
print("\n" + "="*50)
print("OUTDEGREE DISTRIBUTION ACROSS TIME RANGES")
print("="*50)

# Create a new figure for outdegree time-based analysis
plt.figure(figsize=(14, 8))

# Calculate outdegrees for each time range
time_outdegrees = []
for i, (start, end, label) in enumerate(time_ranges):
    # Get nodes in this time range
    nodes_in_range = list(range(start, end + 1))

    # Filter nodes that actually exist in the graph
    existing_nodes = [n for n in nodes_in_range if graph.has_node(n)]

    # Get outdegrees for these nodes
    range_outdegrees = [graph.out_degree(n) for n in existing_nodes]
    time_outdegrees.append(range_outdegrees)

    print(f"{label}: {len(existing_nodes)} nodes, mean outdegree: {np.mean(range_outdegrees):.2f}")

# Find max outdegree across all time ranges for consistent binning
max_outdegree_time = max(max(outdegrees_range)
                         for outdegrees_range in time_outdegrees if outdegrees_range)
bins_time_out = np.arange(0, max_outdegree_time + 2) - 0.5

# Plot the full outdegree distribution as reference (thin black line)
full_out_hist, _ = np.histogram(outdegrees, bins=bins_time_out)
plt.plot(bins_time_out[:-1], full_out_hist, color='black', linewidth=2,
         label='Full Distribution (Reference)', alpha=0.8)

# Plot histograms for each time range
for i, (outdegrees_range, (_, _, label)) in enumerate(zip(time_outdegrees, time_ranges)):
    if outdegrees_range:  # Only plot if we have data
        hist, _ = np.histogram(outdegrees_range, bins=bins_time_out)
        color = colors[i % len(colors)]
        plt.bar(bins_time_out[:-1], hist, width=1, alpha=alphas[i],
                color=color, label=label)

plt.xlabel('Outdegree')
plt.ylabel('Frequency')
plt.title('Outdegree Distribution Across Time Ranges (250k nodes each)')
plt.grid(True, alpha=0.3)
plt.xlim(0, max_outdegree_time + 1)
plt.xticks(np.arange(0, max_outdegree_time + 2, 1))

# Add out-degree bound reference line
plt.axvline(out_degree_bound, color='purple', linestyle='-.',
            linewidth=2, alpha=0.8, label=f'Out-degree bound: {out_degree_bound}')

plt.legend()

# Add statistics
for i, (outdegrees_range, (_, _, label)) in enumerate(zip(time_outdegrees, time_ranges)):
    if outdegrees_range:
        mean_val = np.mean(outdegrees_range)
        color = colors[i % len(colors)]
        plt.axvline(mean_val, color=color, linestyle='--',
                    linewidth=1.5, alpha=0.8)

# Add full distribution mean as reference
full_out_mean = np.mean(outdegrees)
plt.axvline(full_out_mean, color='black', linestyle='-',
            linewidth=2, alpha=0.8, label=f'Full Mean: {full_out_mean:.2f}')

plt.tight_layout()
out_time_path = os.path.join(
    pipeline.output_dir if 'pipeline' in locals() else 'pipeline_outputs',
    f"outdegree_time_distribution_{out_degree_bound}_ef_construction_{ef_construction}_num_nodes_{num_nodes}.png")
plt.savefig(out_time_path, dpi=300, bbox_inches='tight')
plt.show()

# Plot mean degree evolution over time ranges
print("\n" + "="*50)
print("MEAN DEGREE EVOLUTION ACROSS TIME RANGES")
print("="*50)

plt.figure(figsize=(14, 8))

# Calculate means for each time range
range_numbers = []
mean_indegrees_by_range = []
mean_outdegrees_by_range = []
range_labels = []

for i, (indegrees_range, outdegrees_range, (start, end, label)) in enumerate(zip(time_indegrees, time_outdegrees, time_ranges)):
    if indegrees_range and outdegrees_range:
        range_numbers.append(i + 1)
        mean_indegrees_by_range.append(np.mean(indegrees_range))
        mean_outdegrees_by_range.append(np.mean(outdegrees_range))
        range_labels.append(f"Range {i+1}")

# Create the plot
plt.plot(range_numbers, mean_indegrees_by_range, 'o-', color='blue', linewidth=2,
         markersize=8, label='Mean Indegree', alpha=0.8)
plt.plot(range_numbers, mean_outdegrees_by_range, 's-', color='red', linewidth=2,
         markersize=8, label='Mean Outdegree', alpha=0.8)

# Add horizontal reference lines
overall_mean_indegree = np.mean(indegrees)
overall_mean_outdegree = np.mean(outdegrees)
plt.axhline(overall_mean_indegree, color='blue', linestyle='--', alpha=0.6,
            label=f'Overall Mean Indegree: {overall_mean_indegree:.2f}')
plt.axhline(overall_mean_outdegree, color='red', linestyle='--', alpha=0.6,
            label=f'Overall Mean Outdegree: {overall_mean_outdegree:.2f}')
plt.axhline(out_degree_bound, color='green', linestyle='-.', alpha=0.8,
            label=f'Out-degree bound: {out_degree_bound}')

plt.xlabel('Time Range (250k nodes each)')
plt.ylabel('Mean Degree')
plt.title('Evolution of Mean Degree Across Time Ranges')
plt.grid(True, alpha=0.3)
plt.legend()

# Set x-axis ticks to show range numbers
plt.xticks(range_numbers, range_labels, rotation=45)
plt.tight_layout()

# Save the plot
evolution_path = os.path.join(
    pipeline.output_dir if 'pipeline' in locals() else 'pipeline_outputs',
    f"degree_evolution_{out_degree_bound}_ef_construction_{ef_construction}_num_nodes_{num_nodes}.png")
plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
plt.show()

# Print evolution statistics
print("\nMean Degree Evolution Statistics:")
print(f"{'Range':<10} {'Mean In':<12} {'Mean Out':<12} {'In/Out Ratio':<12}")
print("-" * 50)
for i, (mean_in, mean_out) in enumerate(zip(mean_indegrees_by_range, mean_outdegrees_by_range)):
    ratio = mean_in / mean_out if mean_out > 0 else 0
    print(f"Range {i+1:<3} {mean_in:<12.2f} {mean_out:<12.2f} {ratio:<12.2f}")

print(f"\nOverall Statistics:")
print(
    f"Mean Indegree Range: {min(mean_indegrees_by_range):.2f} - {max(mean_indegrees_by_range):.2f}")
print(
    f"Mean Outdegree Range: {min(mean_outdegrees_by_range):.2f} - {max(mean_outdegrees_by_range):.2f}")
print(
    f"Indegree Trend: {'Increasing' if mean_indegrees_by_range[-1] > mean_indegrees_by_range[0] else 'Decreasing'}")
print(
    f"Outdegree Trend: {'Increasing' if mean_outdegrees_by_range[-1] > mean_outdegrees_by_range[0] else 'Decreasing'}")

# Print detailed statistics for each time range
print("\nDetailed Indegree Statistics by Time Range:")
for i, (indegrees_range, (start, end, label)) in enumerate(zip(time_indegrees, time_ranges)):
    if indegrees_range:
        print(f"\n{label} (nodes {start}-{end}):")
        print(f"  Count: {len(indegrees_range)}")
        print(f"  Mean: {np.mean(indegrees_range):.2f}")
        print(f"  Median: {np.median(indegrees_range):.2f}")
        print(f"  Std: {np.std(indegrees_range):.2f}")
        print(f"  Min: {min(indegrees_range)}")
        print(f"  Max: {max(indegrees_range)}")

# Print detailed outdegree statistics for each time range
print("\nDetailed Outdegree Statistics by Time Range:")
for i, (outdegrees_range, (start, end, label)) in enumerate(zip(time_outdegrees, time_ranges)):
    if outdegrees_range:
        print(f"\n{label} (nodes {start}-{end}):")
        print(f"  Count: {len(outdegrees_range)}")
        print(f"  Mean: {np.mean(outdegrees_range):.2f}")
        print(f"  Median: {np.median(outdegrees_range):.2f}")
        print(f"  Std: {np.std(outdegrees_range):.2f}")
        print(f"  Min: {min(outdegrees_range)}")
        print(f"  Max: {max(outdegrees_range)}")
