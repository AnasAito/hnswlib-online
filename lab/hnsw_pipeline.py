"""
HNSW Pipeline for building, visualizing and analyzing HNSW indices.

This module provides a pipeline of steps for:
1. Loading and preprocessing data
2. Building HNSW index
3. Creating adjacency matrix
4. Computing 2D embeddings
5. Visualizing results

Each step saves its output and can be run independently by loading the previous step's data.
"""

from __future__ import annotations
import os
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import hnswlib
import umap
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from matplotlib.legend_handler import HandlerTuple


class HNSWPipeline:
    def __init__(self,
                 data_path: str,
                 output_dir: str = "pipeline_outputs",
                 num_nodes: int = 10_000,
                 out_degree: int = 16,
                 ef_construction: int = 100,
                 seed: int = 123):
        """
        Initialize the HNSW pipeline.

        Args:
            data_path: Path to the input data file (h5 format)
            output_dir: Directory to save pipeline outputs
            num_nodes: Number of nodes to use from the dataset
            out_degree: HNSW M parameter (max number of connections per node)
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.num_nodes = num_nodes
        self.out_degree = out_degree
        self.ef_construction = ef_construction
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize state
        self.data = None
        self.index = None
        self.adj_matrix = None
        self.coords_df = None

    def step1_load_data(self, force_reload: bool = False) -> np.ndarray:
        """
        Step 1: Load and preprocess the data.

        Args:
            force_reload: If True, reload data even if saved version exists

        Returns:
            The loaded and preprocessed data
        """
        output_file = os.path.join(self.output_dir, "processed_data.npy")

        if not force_reload and os.path.exists(output_file):
            print("Loading preprocessed data from cache...")
            self.data = np.load(output_file)
            return self.data

        print("Loading and preprocessing data...")
        with h5py.File(self.data_path, 'r') as f:
            data = f['train_vectors'][:]
            # Sample if needed
            if len(data) > self.num_nodes:
                data = data[self.rng.choice(
                    len(data), self.num_nodes, replace=False)]

        self.data = data
        np.save(output_file, data)
        print(f"Saved processed data to {output_file}")
        return data

    def step2_build_index(self, force_rebuild: bool = False) -> hnswlib.Index:
        """
        Step 2: Build the HNSW index.

        Args:
            force_rebuild: If True, rebuild index even if saved version exists

        Returns:
            The built HNSW index
        """
        if self.data is None:
            self.step1_load_data()

        index_file = os.path.join(self.output_dir, "hnsw_index.bin")

        if not force_rebuild and os.path.exists(index_file):
            print("Loading existing HNSW index...")
            self.index = hnswlib.Index(space='l2', dim=self.data.shape[1])
            self.index.load_index(index_file)
            return self.index

        print("Building HNSW index...")
        self.index = hnswlib.Index(space='l2', dim=self.data.shape[1])
        self.index.init_index(max_elements=self.num_nodes,
                              ef_construction=self.ef_construction,
                              M=self.out_degree)
        self.index.set_num_threads(64)
        self.index.add_items(self.data)

        self.index.save_index(index_file)
        print(f"Saved HNSW index to {index_file}")
        return self.index

    def step3_create_adjacency(self, force_rebuild: bool = False) -> sp.csr_matrix:
        """
        Step 3: Create the adjacency matrix from the HNSW index.

        Args:
            force_rebuild: If True, rebuild adjacency matrix even if saved version exists

        Returns:
            The adjacency matrix in CSR format
        """
        if self.index is None:
            self.step2_build_index()

        adj_file = os.path.join(self.output_dir, "adjacency.npz")

        if not force_rebuild and os.path.exists(adj_file):
            print("Loading existing adjacency matrix...")
            self.adj_matrix = sp.load_npz(adj_file)
            return self.adj_matrix

        print("Creating adjacency matrix...")
        rows, cols, data = [], [], []

        for src in range(self.num_nodes):
            neighbors = self.index.get_neis_list(src)
            for dst in neighbors:
                rows.append(src)
                cols.append(dst)
                data.append(1)

        self.adj_matrix = sp.csr_matrix((data, (rows, cols)),
                                        shape=(self.num_nodes, self.num_nodes),
                                        dtype=np.float32)

        sp.save_npz(adj_file, self.adj_matrix)
        print(f"Saved adjacency matrix to {adj_file}")
        return self.adj_matrix

    def step4_compute_embedding(self,
                                force_recompute: bool = False,
                                umap_neighbors: int = 15,
                                umap_min_dist: float = 0.1) -> pd.DataFrame:
        """
        Step 4: Compute 2D embedding using UMAP.

        Args:
            force_recompute: If True, recompute embedding even if saved version exists
            umap_neighbors: Number of neighbors for UMAP
            umap_min_dist: Minimum distance for UMAP

        Returns:
            DataFrame with node_id, x, y coordinates
        """
        if self.adj_matrix is None:
            self.step3_create_adjacency()

        coords_file = os.path.join(self.output_dir, "coordinates.parquet")

        if not force_recompute and os.path.exists(coords_file):
            print("Loading existing coordinates...")
            self.coords_df = pd.read_parquet(coords_file)
            return self.coords_df

        print("Computing 2D embedding...")
        reducer = umap.UMAP(
            n_components=2,
            metric="euclidean",
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=self.seed,
        )

        coords = reducer.fit_transform(X=self.adj_matrix)

        self.coords_df = pd.DataFrame({
            "node_id": np.arange(self.num_nodes),
            "x": coords[:, 0],
            "y": coords[:, 1]
        })

        self.coords_df.to_parquet(coords_file)
        print(f"Saved coordinates to {coords_file}")
        return self.coords_df

    def step5_visualize(self, query_id: int = None):
        """
        Step 5: Visualize the embedding with optional query point.

        Args:
            query_id: Optional node ID to highlight as query point
        """
        if self.coords_df is None:
            self.step4_compute_embedding()

        plt.figure(figsize=(12, 12), dpi=200)

        if query_id is not None:
            # Compute distances from query point
            query_point = self.data[query_id]
            dist = np.linalg.norm(self.data - query_point, axis=1)

            # Perform search on the index
            self.index.set_ef(10)
            neighbors, distances, traversed_nodes = self.index.knn_query(
                query_point.reshape(1, -1), k=10)

            # --- Draw neighbor edges for traversal nodes (light gray, under points) ---
            for node in traversed_nodes:
                node_neighbors = self.index.get_neis_list(node)
                for neighbor in node_neighbors:
                    if neighbor != node:
                        plt.plot(
                            [self.coords_df['x'][node],
                                self.coords_df['x'][neighbor]],
                            [self.coords_df['y'][node],
                                self.coords_df['y'][neighbor]],
                            color='#bbbbbb', alpha=0.3, linewidth=1, zorder=1
                        )

            # --- Draw all points colored by distance ---
            sc = plt.scatter(
                self.coords_df['x'], self.coords_df['y'],
                c=dist, cmap='plasma', s=30, edgecolor='none', zorder=2
            )

            # --- Highlight query point ---
            plt.scatter(
                self.coords_df['x'][query_id], self.coords_df['y'][query_id],
                c='red', marker='X', s=180, label='Query point', edgecolor='black', linewidths=1.5, zorder=4
            )

            # --- Highlight entry point ---
            if len(traversed_nodes) > 0:
                entry_point = traversed_nodes[0]
                plt.scatter(
                    self.coords_df['x'][entry_point], self.coords_df['y'][entry_point],
                    c='gold', marker='*', s=220, label='Entry point', edgecolor='black', linewidths=1.5, zorder=4
                )

            # --- Draw traversal path with arrows (bold orange) ---
            if len(traversed_nodes) > 1:
                for i in range(len(traversed_nodes) - 1):
                    start_node = traversed_nodes[i]
                    end_node = traversed_nodes[i + 1]
                    plt.arrow(
                        self.coords_df['x'][start_node],
                        self.coords_df['y'][start_node],
                        self.coords_df['x'][end_node] -
                        self.coords_df['x'][start_node],
                        self.coords_df['y'][end_node] -
                        self.coords_df['y'][start_node],
                        head_width=0.12, head_length=0.12,
                        fc='orange', ec='orange', alpha=0.9, width=0.025, zorder=3
                    )

            cbar = plt.colorbar(sc, label='Distance to query',
                                fraction=0.03, pad=0.04)
            cbar.ax.tick_params(labelsize=14)

        else:
            # Plot all points
            plt.scatter(self.coords_df['x'], self.coords_df['y'], s=30)

        plt.title('HNSW Graph Embedding', fontsize=20, pad=20)
        plt.xlabel('UMAP 1', fontsize=16, labelpad=10)
        plt.ylabel('UMAP 2', fontsize=16, labelpad=10)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        if query_id is not None:
            plt.legend(fontsize=14, loc='best', frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'embedding.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    def step5_visualize_density(self, query_id: int = None, hexbin_gridsize: int = 60):
        """
        Visualize the embedding as a density landscape (hexbin) and overlay only the traversal path, query, entry, and neighbors of traversed nodes.
        If query_id is provided, color the hexbin by average distance to the query.
        Traversal nodes and their neighbors are colored by traversal order using a gradient.

        Args:
            query_id: Optional node ID to highlight as query point
            hexbin_gridsize: Controls the resolution of the hexbin density
        """
        import matplotlib as mpl
        from matplotlib import cm
        if self.coords_df is None:
            self.step4_compute_embedding()

        plt.figure(figsize=(12, 12), dpi=200)

        if query_id is not None:
            # --- Compute distances to query ---
            query_point = self.data[query_id]
            distances = np.linalg.norm(self.data - query_point, axis=1)
            # --- Density landscape using hexbin, colored by avg distance ---
            hb = plt.hexbin(
                self.coords_df['x'], self.coords_df['y'],
                C=distances, reduce_C_function=np.mean,
                gridsize=hexbin_gridsize, cmap='PuBu',
                linewidths=0, alpha=0.35, mincnt=1
            )
            cb = plt.colorbar(
                hb, label='Avg. distance to query', fraction=0.03, pad=0.04)
            cb.ax.tick_params(labelsize=14)

            # --- Traversal, query, entry, and neighbors ---
            self.index.set_ef(10)
            neighbors, dists, traversed_nodes = self.index.knn_query(
                query_point.reshape(1, -1), k=10)
            traversed_nodes = list(traversed_nodes)
            traversed_set = set(traversed_nodes)
            n_traversal = len(traversed_nodes)

            # --- Generate a color for each traversal node using a colormap ---
            cmap = cm.get_cmap('viridis', n_traversal)
            traversal_colors = [cmap(i) for i in range(n_traversal)]

            # For colorbar: create a ScalarMappable
            sm = mpl.cm.ScalarMappable(
                cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=n_traversal-1))
            cbar2 = plt.colorbar(
                sm, ticks=[0, n_traversal-1], fraction=0.03, pad=0.01, orientation='vertical')
            cbar2.ax.set_yticklabels(['Start', 'End'])
            cbar2.set_label('Traversal order', fontsize=14)
            cbar2.ax.tick_params(labelsize=13)

            # --- Collect all unique neighbors of traversed nodes (excluding traversal/query) ---
            neighbor_dict = {}  # node idx -> color idx
            for idx, node in enumerate(traversed_nodes):
                node_neighbors = self.index.get_neis_list(node)
                for n in node_neighbors:
                    if n not in traversed_set and n != query_id:
                        # assign color of traversal node
                        neighbor_dict[n] = idx

            # --- Overlay traversal path nodes (colored by order) ---
            for idx, node in enumerate(traversed_nodes):
                plt.scatter(
                    self.coords_df['x'][node], self.coords_df['y'][node],
                    c=[traversal_colors[idx]], marker='o', s=120, label=None, edgecolor='black', linewidths=1.7, zorder=4
                )

            # --- Overlay neighbors of traversed nodes (same color, diamond, lower alpha) ---
            if neighbor_dict:
                for n, color_idx in neighbor_dict.items():
                    plt.scatter(
                        self.coords_df['x'][n], self.coords_df['y'][n],
                        c=[traversal_colors[color_idx]], marker='D', s=80, label=None, edgecolor='black', linewidths=1, alpha=0.5, zorder=3
                    )

            # --- Overlay query point (red X) ---
            plt.scatter(
                self.coords_df['x'][query_id], self.coords_df['y'][query_id],
                c='red', marker='X', s=220, label='Query point', edgecolor='black', linewidths=2.5, zorder=5
            )

            # --- Overlay entry point (gold star) ---
            if len(traversed_nodes) > 0:
                entry_point = traversed_nodes[0]
                plt.scatter(
                    self.coords_df['x'][entry_point], self.coords_df['y'][entry_point],
                    c='gold', marker='*', s=260, label='Entry point', edgecolor='black', linewidths=2.5, zorder=6
                )

            # --- Draw traversal path arrows (colored by order) ---
            if len(traversed_nodes) > 1:
                for i in range(len(traversed_nodes) - 1):
                    start_node = traversed_nodes[i]
                    end_node = traversed_nodes[i + 1]
                    plt.arrow(
                        self.coords_df['x'][start_node],
                        self.coords_df['y'][start_node],
                        self.coords_df['x'][end_node] -
                        self.coords_df['x'][start_node],
                        self.coords_df['y'][end_node] -
                        self.coords_df['y'][start_node],
                        head_width=0.12, head_length=0.12,
                        fc=traversal_colors[i], ec=traversal_colors[i], alpha=0.95, width=0.025, zorder=7
                    )

            # --- Optionally: Draw neighbor edges for each traversed node (same color, faint) ---
            for idx, node in enumerate(traversed_nodes):
                node_neighbors = self.index.get_neis_list(node)
                for neighbor in node_neighbors:
                    if neighbor in neighbor_dict:
                        plt.plot(
                            [self.coords_df['x'][node],
                                self.coords_df['x'][neighbor]],
                            [self.coords_df['y'][node],
                                self.coords_df['y'][neighbor]],
                            color=traversal_colors[idx], alpha=0.13, linewidth=1, zorder=2
                        )
        else:
            # --- Density landscape using hexbin (just density) ---
            hb = plt.hexbin(
                self.coords_df['x'], self.coords_df['y'],
                gridsize=hexbin_gridsize, cmap='PuBu',
                linewidths=0, alpha=0.35, mincnt=1
            )
            cb = plt.colorbar(hb, label='Node density',
                              fraction=0.03, pad=0.04)
            cb.ax.tick_params(labelsize=14)

        plt.title('HNSW Graph Embedding (Density Landscape)',
                  fontsize=20, pad=20)
        plt.xlabel('UMAP 1', fontsize=16, labelpad=10)
        plt.ylabel('UMAP 2', fontsize=16, labelpad=10)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        if query_id is not None:
            plt.legend(['Query point', 'Entry point'],
                       fontsize=14, loc='best', frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir,
                    'embedding_density.png'), dpi=200, bbox_inches='tight')
        plt.close()

    def step5_visualize_traversal_snapshots(self, query_id: int = None, hexbin_gridsize: int = 60):
        """
        For each step in the traversal, plot a snapshot:
        - Density landscape as background
        - All traversed nodes so far (blue)
        - Latest node in traversal (red)
        - Neighbors of latest node (magenta)
        - Traversal arrows up to current step
        - Query and entry points as before
        Each snapshot is saved as embedding_density_step_{i:02d}.png
        """
        import matplotlib as mpl
        if self.coords_df is None:
            self.step4_compute_embedding()
        if query_id is None:
            raise ValueError(
                "query_id must be provided for traversal snapshots visualization.")

        # Compute distances to query for density
        query_point = self.data[query_id]
        distances = np.linalg.norm(self.data - query_point, axis=1)

        # Get traversal path
        self.index.set_ef(20)
        neighbors, dists, traversed_nodes = self.index.knn_query(
            query_point.reshape(1, -1), k=10)
        traversed_nodes = list(traversed_nodes)
        entry_point = traversed_nodes[0] if traversed_nodes else None

        for step in range(1, len(traversed_nodes) + 1):
            plt.figure(figsize=(12, 12), dpi=200)
            # Density landscape
            hb = plt.hexbin(
                self.coords_df['x'], self.coords_df['y'],
                C=distances, reduce_C_function=np.mean,
                gridsize=hexbin_gridsize, cmap='PuBu',
                linewidths=0, alpha=0.35, mincnt=1
            )
            cb = plt.colorbar(
                hb, label='Avg. distance to query', fraction=0.03, pad=0.04)
            cb.ax.tick_params(labelsize=14)

            # Traversed nodes so far (excluding latest)
            if step > 1:
                plt.scatter(
                    self.coords_df['x'][traversed_nodes[:step-1]],
                    self.coords_df['y'][traversed_nodes[:step-1]],
                    c='royalblue', marker='o', s=100, label='Traversed nodes', edgecolor='black', linewidths=1.5, zorder=4
                )

            # Latest node in traversal (red)
            latest_node = traversed_nodes[step-1]
            plt.scatter(
                self.coords_df['x'][latest_node], self.coords_df['y'][latest_node],
                c='red', marker='o', s=140, label='Latest node', edgecolor='black', linewidths=2, zorder=5
            )

            # Neighbors of latest node (magenta)
            latest_neighbors = self.index.get_neis_list(latest_node)
            # Exclude already traversed nodes and query
            latest_neighbors = [
                n for n in latest_neighbors if n not in traversed_nodes[:step] and n != query_id]
            if latest_neighbors:
                plt.scatter(
                    self.coords_df['x'][latest_neighbors], self.coords_df['y'][latest_neighbors],
                    c='magenta', marker='D', s=80, label='Neighbors of latest', edgecolor='black', linewidths=1, alpha=0.7, zorder=3
                )

            # Query point (red X)
            plt.scatter(
                self.coords_df['x'][query_id], self.coords_df['y'][query_id],
                c='red', marker='X', s=220, label='Query point', edgecolor='black', linewidths=2.5, zorder=6
            )

            # Entry point (gold star)
            if entry_point is not None:
                plt.scatter(
                    self.coords_df['x'][entry_point], self.coords_df['y'][entry_point],
                    c='gold', marker='*', s=260, label='Entry point', edgecolor='black', linewidths=2.5, zorder=7
                )

            # Draw traversal arrows up to current step
            if step > 1:
                for i in range(step-1):
                    start_node = traversed_nodes[i]
                    end_node = traversed_nodes[i + 1]
                    plt.arrow(
                        self.coords_df['x'][start_node],
                        self.coords_df['y'][start_node],
                        self.coords_df['x'][end_node] -
                        self.coords_df['x'][start_node],
                        self.coords_df['y'][end_node] -
                        self.coords_df['y'][start_node],
                        head_width=0.12, head_length=0.12,
                        fc='royalblue', ec='royalblue', alpha=0.95, width=0.025, zorder=8
                    )

            plt.title(
                f'HNSW Traversal Snapshot (Step {step}/{len(traversed_nodes)})', fontsize=20, pad=20)
            plt.xlabel('UMAP 1', fontsize=16, labelpad=10)
            plt.ylabel('UMAP 2', fontsize=16, labelpad=10)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=14, loc='best', frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.output_dir, f'embedding_density_step_{step:02d}.png'), dpi=200, bbox_inches='tight')
            plt.close()

    def _stress(self, params, D_ref, D_pair, w_ref=1.0, w_pair=0.0):
        """Objective function (stress) in polar coordinates."""
        n = D_ref.size
        r = params[:n]
        theta = params[n:]
        # distance to reference term
        ref_loss = w_ref * np.sum((r - D_ref) ** 2)

        # 2‑D coordinates implied by params
        xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

        # pairwise stress term
        pred_pair = pdist(xy)
        pair_loss = w_pair * np.sum((pred_pair - D_pair) ** 2)

        return ref_loss + pair_loss

    def _independent_stress(self, params, D_ref, D_pair):
        """
        Independent objective function where:
        - radius is only optimized based on distance to reference
        - theta is only optimized based on pairwise distances
        """
        n = D_ref.size
        r = params[:n]
        theta = params[n:]

        # Distance to reference term (only affects r)
        ref_loss = np.sum((r - D_ref) ** 2)

        # 2D coordinates implied by params
        xy = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

        # Pairwise stress term (only affects theta)
        pred_pair = pdist(xy)
        pair_loss = np.sum((pred_pair - D_pair) ** 2)

        return ref_loss + pair_loss

    def polar_embedding(self, X, ref_vec, w_ref=1.0, w_pair=0.5, fixed_indices=None, fixed_coords=None):
        """
        Compute 2‑D polar coordinates for each row of X so that:
          • radius ≈ distance to ref_vec
          • pairwise distances are preserved (weight w_pair)
          • points with fixed coordinates remain at those coordinates

        Args:
            X: Data matrix to embed
            ref_vec: Reference vector for distances
            w_ref: Weight for reference distance term
            w_pair: Weight for pairwise distance term
            fixed_indices: Indices of points with fixed coordinates (if any)
            fixed_coords: Fixed coordinates for specified indices (if any)

        Returns:
        -------
        coords : (n, 2) array of x, y
        r_opt  : (n,)   optimal radii
        theta_opt : (n,) optimal angles (0–2π)
        stress_val : float  final stress
        """
        n = X.shape[0]
        D_ref = np.linalg.norm(X - ref_vec, axis=1)   # distance to reference
        # pairwise distances in ℝ^d
        D_pair = pdist(X)

        # If we have fixed coordinates
        if fixed_indices is not None and fixed_coords is not None:
            # Determine which indices are free to optimize
            all_indices = np.arange(n)
            free_indices = np.array(
                [i for i in all_indices if i not in fixed_indices])

            if len(free_indices) == 0:
                # All points are fixed, just return the fixed coordinates
                coords = np.zeros((n, 2))
                coords[fixed_indices] = fixed_coords

                # Convert to polar
                r_opt = np.linalg.norm(coords, axis=1)
                theta_opt = np.arctan2(coords[:, 1], coords[:, 0])
                theta_opt = np.mod(theta_opt, 2 * np.pi)

                # Stress value is 0 since we're not optimizing
                return coords, r_opt, theta_opt, 0.0

            # Initial guess: exact radii for free points, fixed for others
            init_r = np.zeros(n)
            init_theta = np.zeros(n)

            # For free points, use regular initialization
            init_r[free_indices] = D_ref[free_indices]

            # Distribute free points evenly around circle
            free_n = len(free_indices)
            init_theta[free_indices] = np.linspace(
                0, 2 * np.pi, free_n, endpoint=False)

            # For fixed points, convert fixed_coords to polar
            fixed_r = np.linalg.norm(fixed_coords, axis=1)
            fixed_theta = np.arctan2(fixed_coords[:, 1], fixed_coords[:, 0])
            fixed_theta = np.mod(fixed_theta, 2 * np.pi)

            init_r[fixed_indices] = fixed_r
            init_theta[fixed_indices] = fixed_theta

            # Create parameter vector (only free parameters)
            init = np.concatenate(
                [init_r[free_indices], init_theta[free_indices]])

            # Define a partial stress function that incorporates fixed points
            def partial_stress(params):
                # Reconstruct full parameter vector
                full_r = init_r.copy()
                full_theta = init_theta.copy()

                # Update free parameters
                n_free = len(free_indices)
                full_r[free_indices] = params[:n_free]
                full_theta[free_indices] = params[n_free:]

                # Convert to cartesian
                full_params = np.concatenate([full_r, full_theta])

                # Compute stress using the full parameter vector
                return self._stress(full_params, D_ref, D_pair, w_ref, w_pair)

            # Optimize only the free parameters
            bounds = [(0, None)] * len(free_indices) + \
                [(None, None)] * len(free_indices)
            res = minimize(
                partial_stress,
                init,
                method="L-BFGS-B",
                bounds=bounds,
            )

            # Reconstruct full parameter vector from optimization result
            r_opt = init_r.copy()
            theta_opt = init_theta.copy()

            r_opt[free_indices] = res.x[:len(free_indices)]
            theta_opt[free_indices] = np.mod(
                res.x[len(free_indices):], 2 * np.pi)

        else:
            # No fixed points, original implementation
            init_r = D_ref.copy()
            init_theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
            init = np.concatenate([init_r, init_theta])

            res = minimize(
                self._stress,
                init,
                args=(D_ref, D_pair, w_ref, w_pair),
                method="L-BFGS-B",
                bounds=[(0, None)] * n + [(None, None)] * n,
            )

            r_opt = res.x[:n]
            theta_opt = np.mod(res.x[n:], 2 * np.pi)

        # Convert polar to cartesian coordinates
        coords = np.column_stack(
            (r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)))

        stress_val = self._stress(np.concatenate(
            [r_opt, theta_opt]), D_ref, D_pair, w_ref, w_pair)
        return coords, r_opt, theta_opt, stress_val

    def polar_embedding_independent(self, X, ref_vec, fixed_indices=None, fixed_coords=None):
        """
        Compute 2D polar coordinates where:
        - radius is optimized ONLY based on distance to reference vector
        - theta is optimized ONLY based on preserving pairwise distances

        This function performs two separate optimization steps:
        1. First optimize r using only reference distances
        2. Then optimize theta using only pairwise distances (with fixed r)

        Args:
            X: Data matrix to embed
            ref_vec: Reference vector for distances
            fixed_indices: Indices of points with fixed coordinates (if any)
            fixed_coords: Fixed coordinates for specified indices (if any)

        Returns:
            coords: (n, 2) array of x, y
            r_opt: (n,) optimal radii
            theta_opt: (n,) optimal angles (0–2π)
            stress_val: float final stress
        """
        n = X.shape[0]
        D_ref = np.linalg.norm(X - ref_vec, axis=1)   # distance to reference
        D_pair = pdist(X)  # pairwise distances in ℝ^d

        # For handling fixed coordinates
        all_indices = np.arange(n)
        if fixed_indices is not None and fixed_coords is not None:
            print("\n==== Optimization with fixed coordinates ====")
            free_indices = np.array(
                [i for i in all_indices if i not in fixed_indices])
            print(
                f"Fixed points: {len(fixed_indices)}, Free points: {len(free_indices)}")

            if len(free_indices) == 0:
                print(
                    "All points are fixed, returning fixed coordinates without optimization")
                # All points are fixed, just return the fixed coordinates
                coords = np.zeros((n, 2))
                coords[fixed_indices] = fixed_coords

                # Convert to polar
                r_opt = np.linalg.norm(coords, axis=1)
                theta_opt = np.arctan2(coords[:, 1], coords[:, 0])
                theta_opt = np.mod(theta_opt, 2 * np.pi)

                return coords, r_opt, theta_opt, 0.0

            # Initialize fixed coordinates
            init_r = np.zeros(n)
            init_theta = np.zeros(n)

            # For fixed points, convert fixed_coords to polar
            fixed_r = np.linalg.norm(fixed_coords, axis=1)
            fixed_theta = np.arctan2(fixed_coords[:, 1], fixed_coords[:, 0])
            fixed_theta = np.mod(fixed_theta, 2 * np.pi)

            init_r[fixed_indices] = fixed_r
            init_theta[fixed_indices] = fixed_theta

            # STEP 1: Optimize r based ONLY on distance to reference
            print("\n--- Step 1: Optimizing radius (r) ---")
            # Calculate initial r loss
            init_r_free = np.zeros(len(free_indices))
            init_r_loss = np.sum((init_r_free - D_ref[free_indices]) ** 2)
            print(f"Initial r loss: {init_r_loss:.4f}")

            # For free points, directly set r to distance from reference (exact solution)
            r_opt = init_r.copy()
            r_opt[free_indices] = D_ref[free_indices]

            # Calculate final r loss
            final_r_loss = np.sum((r_opt - D_ref) ** 2)
            print(f"Final r loss: {final_r_loss:.4f} (exact solution)")

            # STEP 2: Optimize theta using only pairwise distances with fixed r
            print("\n--- Step 2: Optimizing theta ---")
            # Define optimizer for theta only

            def theta_only_stress(theta_params):
                full_theta = init_theta.copy()
                full_theta[free_indices] = theta_params

                # Create 2D coordinates with fixed r
                xy = np.column_stack(
                    (r_opt * np.cos(full_theta), r_opt * np.sin(full_theta)))

                # Pairwise distance stress term
                pred_pair = pdist(xy)
                return np.sum((pred_pair - D_pair) ** 2)

            # Initial guess for theta
            init_theta_free = np.linspace(
                0, 2 * np.pi, len(free_indices), endpoint=False)

            # Calculate initial theta loss
            init_theta_loss = theta_only_stress(init_theta_free)
            print(f"Initial theta loss: {init_theta_loss:.4f}")

            # Optimize theta
            res = minimize(
                theta_only_stress,
                init_theta_free,
                method="L-BFGS-B",
                bounds=[(None, None)] * len(free_indices),
            )

            # Update theta with optimized values
            theta_opt = init_theta.copy()
            theta_opt[free_indices] = np.mod(res.x, 2 * np.pi)

            # Calculate final theta loss
            final_theta_loss = theta_only_stress(res.x)
            print(f"Final theta loss: {final_theta_loss:.4f}")
            print(
                f"Theta optimization success: {res.success}, iterations: {res.nit}")

        else:
            print("\n==== Optimization without fixed coordinates ====")
            # No fixed points, truly independent implementation

            # STEP 1: Optimize r - simply set to exact distance from reference
            print("\n--- Step 1: Optimizing radius (r) ---")
            # Initial r (all zeros as a baseline)
            init_r = np.zeros(n)
            init_r_loss = np.sum((init_r - D_ref) ** 2)
            print(f"Initial r loss (with zeros): {init_r_loss:.4f}")

            r_opt = D_ref.copy()  # This is the exact solution for r

            # Calculate final r loss (should be zero)
            final_r_loss = np.sum((r_opt - D_ref) ** 2)
            print(f"Final r loss: {final_r_loss:.4f} (exact solution)")

            # STEP 2: Optimize theta using only pairwise distances
            print("\n--- Step 2: Optimizing theta ---")

            def theta_only_stress(theta_params):
                # Create 2D coordinates with fixed r
                xy = np.column_stack(
                    (r_opt * np.cos(theta_params), r_opt * np.sin(theta_params)))

                # Pairwise distance stress term
                pred_pair = pdist(xy)
                return np.sum((pred_pair - D_pair) ** 2)

            # Initial guess for theta - evenly spaced around circle
            init_theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

            # Calculate initial theta loss
            init_theta_loss = theta_only_stress(init_theta)
            print(f"Initial theta loss: {init_theta_loss:.4f}")

            # Optimize theta
            res = minimize(
                theta_only_stress,
                init_theta,
                method="L-BFGS-B",
                bounds=[(None, None)] * n,
            )

            theta_opt = np.mod(res.x, 2 * np.pi)

            # Calculate final theta loss
            final_theta_loss = theta_only_stress(theta_opt)
            print(f"Final theta loss: {final_theta_loss:.4f}")
            print(
                f"Theta optimization success: {res.success}, iterations: {res.nit}")

        # Convert polar to cartesian coordinates
        coords = np.column_stack(
            (r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)))

        # Calculate final stress
        xy = np.column_stack(
            (r_opt * np.cos(theta_opt), r_opt * np.sin(theta_opt)))
        pred_pair = pdist(xy)
        ref_loss = np.sum((r_opt - D_ref) ** 2)
        pair_loss = np.sum((pred_pair - D_pair) ** 2)
        stress_val = ref_loss + pair_loss

        print("\n--- Final combined stress ---")
        print(f"Reference loss: {ref_loss:.4f}")
        print(f"Pairwise loss: {pair_loss:.4f}")
        print(f"Total stress: {stress_val:.4f}")

        return coords, r_opt, theta_opt, stress_val

    def step5_visualize_traversal_dim_reduce(self, query_id: int = None, ef_search: int = 10):
        """
        Visualize the traversal using polar embedding for dimensionality reduction.
        For each step in the traversal, creates a snapshot where:
        - The query point is used as the reference vector for polar embedding
        - Only the current traversal node and its neighbors are shown
        - Coordinates are computed to preserve distances from query and between points
        - Traversal arrows show the path taken
        - Previously seen points maintain fixed positions across snapshots
        - Consistent scale is maintained across all snapshots
        - Neighbors from previous steps shown with fading opacity
        - Neighbors that have been seen in previous traversal steps are colored differently

        Args:
            query_id: Node ID to use as query point (required)
        """
        if query_id is None:
            raise ValueError(
                "query_id must be provided for traversal visualization.")

        if self.index is None:
            self.step2_build_index()

        # Get query point vector
        query_point = self.data[query_id]

        # Get traversal path
        self.index.set_ef(ef_search)
        neighbors, dists, traversed_nodes, res_queues = self.index.knn_query(
            query_point.reshape(1, -1), k=10)
        print(f"res_queues: {res_queues.shape}")
        # decompose res_queues into list of lists
        queue = []
        queue_list = []
        for node in res_queues:
            if node == self.data.shape[0] + 100:
                queue_list.append(queue)
                queue = []
            else:
                queue.append(node)
        if queue:
            queue_list.append(queue)
        print(f"number of queues: {len(queue_list)}")
        print(f"queue_list: {queue_list}")

        # compute true neighbors with query
        query_dists = np.linalg.norm(self.data - query_point, axis=1)
        true_neighbors = np.argsort(query_dists)[:10]
        recalls = []
        for queue in queue_list:
            # compute current recall
            current_recall = len(set(queue) & set(
                true_neighbors)) / len(true_neighbors)
            recalls.append(current_recall)
            print(f"current recall: {current_recall}")

        traversed_nodes = list(traversed_nodes)
        entry_point = traversed_nodes[0] if traversed_nodes else None

        # Track all seen nodes and their coordinates
        all_seen_nodes = {query_id}  # We always include the query
        node_coordinates = {}  # Will map node_id -> (x, y) coordinates

        # Track neighbors seen at each step
        neighbors_by_step = {}  # step -> list of neighbor node_ids

        # Track all neighbors we've ever seen during traversal
        all_previously_seen_neighbors = set()

        # Set the query point at the origin
        node_coordinates[query_id] = np.array([0.0, 0.0])

        # First pass to compute all coordinates and determine global maximum radius
        global_max_radius = 0.0
        all_radii = []

        print("Computing coordinates for all steps...")
        for step in range(1, len(traversed_nodes) + 1):
            # Get nodes to visualize for this step
            latest_node = traversed_nodes[step-1]
            latest_neighbors = self.index.get_neis_list(latest_node)
            print(f"Latest node: {latest_node}, neighbors: {latest_neighbors}")
            filtered_neighbors = [n for n in latest_neighbors
                                  if n not in traversed_nodes[:step-1] and n != latest_node]

            # Store neighbors for this step
            neighbors_by_step[step] = filtered_neighbors

            vis_node_ids = traversed_nodes[:step] + filtered_neighbors
            if query_id not in vis_node_ids:
                vis_node_ids.append(query_id)

            all_seen_nodes.update(vis_node_ids)
            vis_node_ids = np.array(vis_node_ids).astype(int)
            vis_data = self.data[vis_node_ids]

            # Determine which nodes already have fixed coordinates
            fixed_nodes = []
            fixed_coords = []
            for i, node_id in enumerate(vis_node_ids):
                if node_id in node_coordinates:
                    fixed_nodes.append(i)
                    fixed_coords.append(node_coordinates[node_id])

            if fixed_nodes:
                fixed_nodes = np.array(fixed_nodes)
                fixed_coords = np.array(fixed_coords)
            else:
                fixed_nodes = None
                fixed_coords = None

            # Use polar embedding to compute 2D coordinates with fixed points
            coords2d, radii, angles, _ = self.polar_embedding_independent(
                vis_data, query_point,
                # fixed_indices=fixed_nodes,
                # fixed_coords=fixed_coords
            )

            # Store computed coordinates for future steps
            for i, node_id in enumerate(vis_node_ids):
                if node_id not in node_coordinates:
                    node_coordinates[node_id] = coords2d[i]

            # Update global maximum radius
            max_step_radius = radii.max()
            all_radii.append(max_step_radius)
            global_max_radius = max(global_max_radius, max_step_radius)

            # Update set of all neighbors we've seen so far
            if step > 1:
                # Add neighbors from previous steps to the set of previously seen neighbors
                for prev_step in range(1, step):
                    prev_neighbors = neighbors_by_step.get(prev_step, [])
                    all_previously_seen_neighbors.update(prev_neighbors)

        # Add a margin to the global maximum radius for better visualization
        global_max_radius *= 1.2

        print(f"Global maximum radius: {global_max_radius}")

        # Reset the previously seen neighbors tracking for the visualization pass
        all_previously_seen_neighbors = set()

        # Second pass to create visualizations with consistent scaling
        print("Creating visualizations...")
        for step in range(1, len(traversed_nodes) + 1):
            plt.figure(figsize=(12, 12), dpi=200)

            # Get nodes to visualize
            latest_node = traversed_nodes[step-1]
            latest_neighbors = self.index.get_neis_list(latest_node)
            filtered_neighbors = [n for n in latest_neighbors
                                  if n not in traversed_nodes[:step-1] and n != latest_node]

            # Divide neighbors into new ones and previously seen ones
            previously_seen_filtered = [
                n for n in filtered_neighbors if n in all_previously_seen_neighbors]
            new_filtered = [
                n for n in filtered_neighbors if n not in all_previously_seen_neighbors]

            vis_node_ids = traversed_nodes[:step] + filtered_neighbors
            if query_id not in vis_node_ids:
                vis_node_ids.append(query_id)

            # Convert to array for indexing
            vis_node_ids = np.array(vis_node_ids).astype(int)

            # Get coordinates for these nodes from our pre-computed dictionary
            coords2d = np.array([node_coordinates[node_id]
                                for node_id in vis_node_ids])

            # Calculate radii for these points (distance from origin)
            radii = np.linalg.norm(coords2d, axis=1)

            # Create mapping from node_id to position in coords2d
            node_to_pos = {node_id: i for i,
                           node_id in enumerate(vis_node_ids)}

            # Plot query point (center)
            query_pos = node_to_pos[query_id]
            plt.scatter(
                coords2d[query_pos, 0], coords2d[query_pos, 1],
                c='red', marker='X', s=220, label='Query point',
                edgecolor='black', linewidths=2.5, zorder=6
            )

            # Plot entry point
            if entry_point is not None and entry_point in node_to_pos:
                entry_pos = node_to_pos[entry_point]
                plt.scatter(
                    coords2d[entry_pos, 0], coords2d[entry_pos, 1],
                    c='gold', marker='*', s=260, label='Entry point',
                    edgecolor='black', linewidths=2.5, zorder=7
                )

            # Plot traversed nodes so far (excluding latest)
            if step > 1:
                for node in traversed_nodes[:step-1]:
                    if node in node_to_pos:
                        pos = node_to_pos[node]
                        plt.scatter(
                            coords2d[pos, 0], coords2d[pos, 1],
                            c='royalblue', marker='o', s=100,
                            edgecolor='black', linewidths=1.5, zorder=4
                        )

            # Plot latest node in traversal
            latest_pos = node_to_pos[latest_node]
            plt.scatter(
                coords2d[latest_pos, 0], coords2d[latest_pos, 1],
                c='red', marker='o', s=140, label='Latest node',
                edgecolor='black', linewidths=2, zorder=5
            )

            # Plot queue nodes for current step (as red circles)
            if step - 1 < len(queue_list):
                current_queue = queue_list[step - 1]
                for node in current_queue:
                    # Skip if node is already handled (like the latest node or entry point)
                    if node in node_coordinates and node != latest_node and node != query_id:
                        coords = node_coordinates[node]
                        # First plot as purple diamond (like neighbors)
                        plt.scatter(
                            coords[0], coords[1],
                            c='magenta', marker='D', s=80,
                            edgecolor='black', linewidths=1,
                            alpha=0.7, zorder=3
                        )
                        # Then overlay a small red circle to indicate it's in queue
                        plt.scatter(
                            coords[0], coords[1],
                            c='black', marker='o', s=25,
                            edgecolor='black', linewidths=0.5,
                            alpha=1, zorder=5
                        )
                # Add legend entry for queue nodes
                if current_queue:
                    # Create custom legend entry showing diamond with circle
                    legend_diamond = plt.scatter([], [], c='magenta', marker='D', s=80,
                                                 edgecolor='black', linewidths=1, alpha=0.7)
                    legend_circle = plt.scatter([], [], c='black', marker='o', s=25,
                                                edgecolor='black', linewidths=0.5, alpha=1)
                    plt.legend([(legend_diamond, legend_circle)], ['Queue nodes'],
                               handler_map={tuple: HandlerTuple(ndivide=None)},
                               fontsize=14, loc='best', frameon=True)

            # Plot new neighbors of latest node
            for node in new_filtered:
                if node != query_id and node in node_to_pos:  # Skip query if it's a neighbor
                    pos = node_to_pos[node]
                    plt.scatter(
                        coords2d[pos, 0], coords2d[pos, 1],
                        c='magenta', marker='D', s=80, label='_nolegend_',
                        edgecolor='black', linewidths=1, alpha=0.7, zorder=3
                    )

            # Plot previously seen neighbors of latest node (with different color)
            for node in previously_seen_filtered:
                if node != query_id and node in node_to_pos:
                    pos = node_to_pos[node]
                    plt.scatter(
                        coords2d[pos, 0], coords2d[pos, 1],
                        c='cyan', marker='D', s=80, label='_nolegend_',
                        edgecolor='black', linewidths=1, alpha=0.7, zorder=3
                    )

            # Plot neighbors from previous steps with fading opacity
            if step > 1:
                # Collect all previous neighbors
                all_prev_neighbors = set()
                for prev_step in range(1, step):
                    # Calculate opacity based on recency
                    # More recent = higher opacity
                    opacity = 0.8 * (prev_step / step)

                    # Get neighbors for this previous step
                    prev_neighbors = neighbors_by_step.get(prev_step, [])

                    # Plot each neighbor that's not already handled
                    for node in prev_neighbors:
                        # Skip if already in traversed path or current neighbors
                        if (node in traversed_nodes[:step] or
                            node == query_id or
                            node in filtered_neighbors or
                                node in all_prev_neighbors):  # Skip if already plotted from another step
                            continue

                        # Only plot if we have coordinates
                        if node in node_coordinates:
                            all_prev_neighbors.add(node)
                            coords = node_coordinates[node]
                            plt.scatter(
                                coords[0], coords[1],
                                c='magenta', marker='D', s=50,
                                edgecolor='grey', linewidths=0.5,
                                alpha=opacity, zorder=2
                            )

            # Add legend entries for neighbors
            if new_filtered:
                plt.scatter([], [], c='magenta', marker='D', s=80,
                            label='New neighbors', edgecolor='black',
                            linewidths=1, alpha=0.7)

            if previously_seen_filtered:
                plt.scatter([], [], c='cyan', marker='D', s=80,
                            label='Previously seen neighbors', edgecolor='black',
                            linewidths=1, alpha=0.7)

            # Add legend entry for previous neighbors if we have any
            if step > 1:
                plt.scatter([], [], c='magenta', marker='D', s=50,
                            label='Neighbors from previous steps', edgecolor='grey',
                            linewidths=0.5, alpha=0.4)

            # Draw traversal arrows up to current step
            if step > 1:
                for i in range(step-1):
                    start_node = traversed_nodes[i]
                    end_node = traversed_nodes[i + 1]
                    if start_node in node_to_pos and end_node in node_to_pos:
                        start_pos = node_to_pos[start_node]
                        end_pos = node_to_pos[end_node]
                        plt.arrow(
                            coords2d[start_pos, 0], coords2d[start_pos, 1],
                            coords2d[end_pos, 0] - coords2d[start_pos, 0],
                            coords2d[end_pos, 1] - coords2d[start_pos, 1],
                            head_width=global_max_radius * 0.04,
                            head_length=global_max_radius * 0.04,
                            length_includes_head=True,
                            fc='royalblue', ec='royalblue', alpha=0.95,
                            width=global_max_radius * 0.005, zorder=8
                        )

            # Add concentric circles to indicate distance from query
            # Use consistent global max radius for all plots
            circle_radii = np.linspace(
                global_max_radius * 0.25, global_max_radius * 0.9, 3)
            for r in circle_radii:
                circle = plt.Circle((0, 0), r, fill=False,
                                    linestyle='--', color='gray', alpha=0.5)
                plt.gca().add_patch(circle)

            # Add radial lines
            angles_lines = np.linspace(0, 2*np.pi, 8, endpoint=False)
            for angle in angles_lines:
                dx = global_max_radius * np.cos(angle)
                dy = global_max_radius * np.sin(angle)
                plt.plot([0, dx], [0, dy],
                         linestyle='--', color='gray', alpha=0.5)

            # Set consistent plot limits for all steps
            plt.xlim(-global_max_radius, global_max_radius)
            plt.ylim(-global_max_radius, global_max_radius)

            plt.title(
                f'HNSW Traversal (Step {step}/{len(traversed_nodes)})\nCurrent Recall: {recalls[step-1]:.2f}',
                fontsize=20, pad=20
            )

            # Ensure equal aspect ratio
            plt.axis('equal')

            # Remove axes for cleaner visualization
            plt.axis('off')

            plt.legend(fontsize=14, loc='best', frameon=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir,
                             f'traversal_polar_step_{step:02d}.png'),
                dpi=200, bbox_inches='tight'
            )
            plt.close()

            # After visualization, update the set of all previously seen neighbors
            all_previously_seen_neighbors.update(filtered_neighbors)

    def run_pipeline(self, start_step: int = 1, end_step: int = 5, **kwargs):
        """
        Run the pipeline from start_step to end_step.

        Args:
            start_step: First step to run (1-5)
            end_step: Last step to run (1-5)
            **kwargs: Additional arguments for specific steps
        """
        steps = {
            1: self.step1_load_data,
            2: self.step2_build_index,
            # 3: self.step3_create_adjacency,
            # 4: self.step4_compute_embedding,
            3: self.step5_visualize_traversal_dim_reduce
        }

        for step_num in range(start_step, end_step + 1):
            print(f"\nRunning step {step_num}...")
            steps[step_num](**kwargs)
