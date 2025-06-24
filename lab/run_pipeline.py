"""
Example script showing how to use the HNSW pipeline.
"""

import os
import shutil
from hnsw_pipeline import HNSWPipeline

# rmv ouput dir if exists
if os.path.exists("pipeline_outputs"):
    shutil.rmtree("pipeline_outputs")


def main():

    # Initialize pipeline
    pipeline = HNSWPipeline(
        data_path='/data/anas.aitaomar/sift_1m_old_dist.h5',
        # data_path='/home/anas.aitaomar/yfcc/yfcc_10m_old_dist.h5',
        output_dir='pipeline_outputs',
        num_nodes=1_000_000,
        out_degree=4,
        ef_construction=100,
        seed=None
    )

    # Example 1: Run full pipeline

    # print("\nRunning full pipeline...")
    try:
        pipeline.run_pipeline()
    except Exception as e:
        print(e)

    # Example 2: Run specific steps
    print("\nRunning only visualization step with query point...")
    pipeline.run_pipeline(start_step=3, end_step=3,
                          query_id=53421, ef_search=10)

    # generate gif
    import imageio.v3 as iio
    import os

    output_dir = pipeline.output_dir
    image_files = sorted([
        os.path.join(output_dir, fname)
        for fname in os.listdir(output_dir)
        if fname.startswith("traversal_polar_step_") and fname.endswith(".png")
    ])

    frames = [iio.imread(img) for img in image_files]
    iio.imwrite(f"{output_dir}/traversal.gif", frames, duration=1000, loop=0)


if __name__ == "__main__":
    main()
