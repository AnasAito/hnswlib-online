# graph2map_fast.py
"""
Fast 2-D embedding + diagnostics for (possibly directed) HNSW base graphs.

Key ideas
---------
1.  Use UMAP in 'graph' mode → no O(n²) distance matrix.
2.  Accept scipy.sparse CSR adjacency; works for 100k+ nodes.
3.  Trustworthiness / continuity / k-NN recall are estimated on a
    random sample (configurable) so debug mode stays quick.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import hnswlib
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import umap
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness

# -------------------------------------------------------------------
# --- helpers --------------------------------------------------------
# -------------------------------------------------------------------


def _sample_rows(mat: sp.spmatrix, m: int, rng) -> sp.spmatrix:
    idx = rng.choice(mat.shape[0], size=m, replace=False)
    return mat[idx][:, idx], idx


def _knn_recall_sparse(adj_high: sp.spmatrix,
                       coords_low: np.ndarray,
                       k: int = 10) -> float:
    """
    Edge recall between the sparse high-D adjacency and the Euclidean
    k-NN graph in 2-D.
    """
    # high-D: neighbours present in adjacency
    high_sets = [set(adj_high[i].indices) for i in range(adj_high.shape[0])]
    # low-D: k nearest by Euclidean distance
    d_low = pairwise_distances(coords_low, metric="euclidean")
    low_nn = np.argsort(d_low, axis=1)[:, 1:k+1]
    hits = sum(len(high_sets[i].intersection(low_nn[i]))
               for i in range(adj_high.shape[0]))
    total = k * adj_high.shape[0]
    return hits / total

# -------------------------------------------------------------------


def graph_to_2d_fast(
    adj: sp.spmatrix,
    *,
    mode: str = "map",           # 'map' or 'debug'
    make_undirected: bool = True,
    sample_debug: int = 2000,    # nodes to sample for metrics
    k_metrics: int = 10,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int | None = 42,
):
    """
    Parameters
    ----------
    adj : scipy.sparse matrix  (n × n, CSR preferred)
        HNSW base-layer adjacency. Non-zero values are edge weights
        (e.g. original distances). Zeros = no edge.
    make_undirected : bool
        True → add transpose and keep min(weight, weight^T).
        False → leave directed.
    mode : 'map' or 'debug'
        'map'  → DataFrame with node_id, x, y
        'debug' → dict with {stress, trustworthiness, continuity, knn_recall}
    sample_debug : int
        Number of nodes to subsample when computing diagnostics.
        Ignored in 'map' mode.
    """
    rng = check_random_state(random_state)

    # 1. sanitise / symmetrise if requested
    if make_undirected:
        adj = adj.minimum(adj.T) + adj.maximum(adj.T)  # keep both weights
        adj = adj.tocsr()

    # 2. run UMAP directly on the sparse graph
    reducer = umap.UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )

    coords = reducer.fit_transform(X=adj)

    if mode == "map":
        return pd.DataFrame(
            {"node_id": np.arange(adj.shape[0]),
             "x": coords[:, 0], "y": coords[:, 1]}
        )

    # ------------------  debug mode  ------------------
    # subsample to keep O(sample²) work small
    adj_s, idx = _sample_rows(adj, min(sample_debug, adj.shape[0]), rng)
    coords_s = coords[idx]

    # trustworthiness (sklearn wants original data OR distances)
    d_high_sub = pairwise_distances(adj_s.toarray(), metric="euclidean")
    d_low_sub = pairwise_distances(coords_s, metric="euclidean")

    # Kruskal stress on the sample
    stress = np.sqrt(((d_high_sub - d_low_sub) ** 2).sum() /
                     (d_high_sub ** 2).sum())

    # trustworthiness / continuity
    tw = trustworthiness(d_high_sub, coords_s,
                         metric="precomputed", n_neighbors=k_metrics)
    # continuity (quick sample-based variant)
    ranks_high = np.argsort(d_high_sub, axis=1)
    ranks_low = np.argsort(d_low_sub, axis=1)
    cont_hits = 0
    for i in range(ranks_high.shape[0]):
        cont_hits += len(set(ranks_high[i, 1:k_metrics+1])
                         .intersection(ranks_low[i, 1:k_metrics+1]))
    continuity = cont_hits / (k_metrics * ranks_high.shape[0])

    # k-NN edge recall (on full graph, but cheap)
    knn_rec = _knn_recall_sparse(adj, coords, k=k_metrics)

    return {
        "stress": stress,
        "trustworthiness": tw,
        "continuity": continuity,
        "knn_recall": knn_rec,
        "sample_size": adj_s.shape[0],
    }


# ------------------------------------------------------------------
# 1. CONFIG  --------------------------------------------------------
# ------------------------------------------------------------------
NUM_NODES = 10_000          # size of your base layer
OUT_DEGREE = 16              # typical HNSW M parameter
SEED = 123             # for reproducibility
OUT_FILE = "hnsw_base_mock.npz"
# ------------------------------------------------------------------
# 2. CREATE  THE INDEX  ------------------------------------------------
# ------------------------------------------------------------------


# Load the SIFT dataset from the h5 file
print("Loading SIFT dataset from h5 file...")
with h5py.File('/data/anas.aitaomar/sift_1m_old_dist.h5', 'r') as f:
    sift_data = f['train_vectors'][:]
    # sample
    sift_data_sample = sift_data[np.random.randint(
        0, sift_data.shape[0], size=NUM_NODES)]
    print(f"Loaded SIFT data with shape: {sift_data_sample.shape}")


# build index
p = hnswlib.Index(space='l2', dim=sift_data_sample.shape[1])
p.init_index(max_elements=NUM_NODES, ef_construction=100, M=16)
p.set_num_threads(64)
p.add_items(sift_data_sample)
#!/usr/bin/env python
"""
demo_build_and_embed.py
------------------------------------------------------------
Mocks an HNSW-like 'get_out_neighbors' API, constructs a sparse
adjacency matrix, saves it, and then calls graph_to_2d_fast().
"""


rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# 2. MOCK THE INDEX  ------------------------------------------------
# ------------------------------------------------------------------


def get_out_neighbors(node_id: int, k: int = OUT_DEGREE):
    """
    Placeholder for your real HNSW query:
        returns a list of (dst_id, distance)
    Here we just pick k distinct random nodes and invent a distance.
    """
    return p.get_neis_list(node_id)


# ------------------------------------------------------------------
# 3. BUILD THE SPARSE MATRIX  --------------------------------------
# ------------------------------------------------------------------
rows, cols, data = [], [], []

for src in range(NUM_NODES):
    for dst in get_out_neighbors(src):
        rows.append(src)
        cols.append(dst)
        data.append(1)

adj_csr = sp.csr_matrix((data, (rows, cols)),
                        shape=(NUM_NODES, NUM_NODES),
                        dtype=np.float32)

sp.save_npz(OUT_FILE, adj_csr)
print(f"Saved adjacency to {OUT_FILE}  "
      f"({adj_csr.nnz:,} directed edges, density={adj_csr.nnz/NUM_NODES**2:.2e})")

# ------------------------------------------------------------------
# 4. EMBED + DIAGNOSE  ---------------------------------------------
# ------------------------------------------------------------------
# (keep it directed for a realistic test)
# load the adjacency matrix
adj_csr = sp.load_npz(OUT_FILE)

coords_df = graph_to_2d_fast(adj_csr,
                             mode="map",
                             make_undirected=False,
                             umap_neighbors=15,
                             umap_min_dist=0.1,
                             random_state=None)

coords_df.to_parquet("coords_mock.parquet")
print("Embedded → coords_mock.parquet  (first rows):")
print(coords_df.head())

metrics = graph_to_2d_fast(adj_csr,
                           mode="debug",
                           make_undirected=False,
                           sample_debug=2000,
                           k_metrics=10,
                           random_state=SEED)

print("\nQuality metrics on 2 000-node sample:")
for k, v in metrics.items():
    print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k:15s}: {v}")
# plot coords
plt.scatter(coords_df['x'], coords_df['y'])
plt.show()
# ------------------------------------------------------------------
# 5. Compute distances against a given query  ------------------------------------------------
# ------------------------------------------------------------------
query_id = 0
query_point = sift_data_sample[query_id]
# compute l2 distance
dist = np.linalg.norm(sift_data_sample - query_point, axis=1)

# use dist as color (log) + show legend
plt.scatter(coords_df['x'], coords_df['y'], c=np.log(dist))
# color query point in red
plt.scatter(coords_df['x'][query_id], coords_df['y']
            [query_id], c='red', marker='x')
plt.colorbar(label='log distance')
plt.show()
