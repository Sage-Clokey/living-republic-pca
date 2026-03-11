"""
run_analysis.py
---------------
End-to-end pipeline:
  1. Download all datasets
  2. Run congress PCA + clustering + all plots
  3. Run UN PCA + clustering + all plots
  4. Save every output to outputs/

Usage:
    python src/run_analysis.py
"""

from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── imports ───────────────────────────────────────────────────────────────────
from download_data import download_all, verify_downloads
from voting_pca import (
    # Congress — original
    load_congress_data,
    build_vote_matrix,
    run_pca,
    run_clustering,
    silhouette_analysis,
    plot_pca_with_party_labels,
    plot_natural_clusters_vs_party,
    plot_variance_explained,
    plot_crossover_members,
    plot_party_cluster_alignment,
    plot_silhouette,
    make_interactive_scatter,
    # Congress — temporal spiral
    plot_temporal_helix,
    plot_party_strands,
    plot_congress_colored,
    plot_3d_helix,
    # Congress — Sagent / Spiral-and-Block themed
    plot_the_jar,
    plot_spiral_vs_block,
    plot_jothams_parable,
    plot_free_market_of_law,
    plot_living_dimensions,
    # UN
    load_un_data,
    build_un_vote_matrix,
    plot_un_country_clusters,
    make_un_interactive,
)


def banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_congress_analysis() -> None:
    banner("ANALYSIS 1 — CONGRESS VOTING PATTERNS")
    t0 = time.time()

    # 1. Load
    members, votes, rollcalls = load_congress_data(
        congress_range=(110, 118), chamber="House"
    )

    # 2. Build vote matrix
    vote_matrix, meta = build_vote_matrix(votes, members)

    # 3. PCA
    pca_coords, pca_model, var_ratio = run_pca(vote_matrix, n_components=10)

    # 4. Clustering
    k_range = [2, 3, 4, 5, 6, 8, 10]
    cluster_results = run_clustering(pca_coords, k_range=k_range)

    # 5. Silhouette
    print("\nComputing silhouette scores …")
    sil_scores = silhouette_analysis(pca_coords, k_range=k_range)
    print("Silhouette scores:", {k: f"{v:.4f}" for k, v in sil_scores.items()})

    # 6. Plots — original analysis
    print("\nGenerating base analysis plots …")
    plot_variance_explained(var_ratio)
    plot_pca_with_party_labels(pca_coords, meta)
    plot_natural_clusters_vs_party(pca_coords, cluster_results, meta, ks=[3, 4, 5])
    plot_party_cluster_alignment(cluster_results, meta, k=3)
    plot_silhouette(sil_scores)
    plot_crossover_members(pca_coords, meta)
    make_interactive_scatter(pca_coords, meta)

    # 7. Plots — Sagent / Spiral-and-Block thesis visualizations
    # Use the best natural k (prefer 4 or 5 if available, else 3)
    best_k = max(sil_scores, key=sil_scores.get)
    natural_labels = cluster_results[best_k]
    # 7. Plots — temporal spiral (the DNA crystallography connection)
    print("\nGenerating temporal spiral plots …")
    plot_congress_colored(pca_coords, meta)
    plot_party_strands(pca_coords, meta)
    plot_temporal_helix(pca_coords, meta)
    plot_3d_helix(pca_coords, meta)

    print(f"\nGenerating Sagent-themed thesis plots (best natural k={best_k}) …")
    plot_the_jar(pca_coords, meta, natural_labels)
    plot_spiral_vs_block(pca_coords, meta)
    plot_jothams_parable(pca_coords, meta, natural_labels)
    plot_free_market_of_law(pca_coords, meta, natural_labels)
    plot_living_dimensions(var_ratio)

    print(f"\nCongress analysis complete in {time.time()-t0:.1f}s")


def run_un_analysis() -> None:
    banner("ANALYSIS 2 — UN GENERAL ASSEMBLY VOTING")
    t0 = time.time()

    # 1. Load
    un_votes, un_rc, un_issues = load_un_data()

    # 2. Build matrix
    un_matrix, blocks = build_un_vote_matrix(un_votes)

    # 3. PCA
    pca_coords, pca_model, var_ratio = run_pca(un_matrix, n_components=10)

    # 4. Clustering
    cluster_results = run_clustering(pca_coords, k_range=[2, 3, 4, 5, 6])

    # 5. Silhouette
    sil = silhouette_analysis(pca_coords, k_range=[2, 3, 4, 5, 6])
    print("UN Silhouette scores:", {k: f"{v:.4f}" for k, v in sil.items()})

    # 6. Plots
    print("\nGenerating UN plots …")
    plot_un_country_clusters(
        pca_coords,
        un_matrix.index,
        blocks,
        cluster_labels=cluster_results.get(4),
    )
    make_un_interactive(pca_coords, un_matrix.index, blocks)

    print(f"\nUN analysis complete in {time.time()-t0:.1f}s")


def main() -> None:
    banner("PCA LIVING SYSTEMS — FULL PIPELINE")
    print(f"Project root: {PROJECT_ROOT}")

    # Step 1: Download data
    banner("DOWNLOADING DATA")
    download_all(force=False)
    if not verify_downloads():
        print("ERROR: Some downloads failed.  Check your internet connection.")
        sys.exit(1)

    # Step 2: Congress analysis
    run_congress_analysis()

    # Step 3: UN analysis
    run_un_analysis()

    banner("PIPELINE COMPLETE")
    out = PROJECT_ROOT / "outputs"
    files = sorted(out.iterdir())
    print(f"\nOutputs saved to {out}/")
    for f in files:
        size = f.stat().st_size / 1e3
        print(f"  {f.name:50s}  {size:>8.1f} KB")


if __name__ == "__main__":
    main()
