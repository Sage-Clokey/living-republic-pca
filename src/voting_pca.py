"""
voting_pca.py
-------------
Reusable analysis functions for the "Imposed Binary vs. Natural Structure"
project.  All heavy logic lives here so that notebooks stay readable.

Key functions
─────────────
Congress analysis
  load_congress_data(congress_range)
  build_vote_matrix(votes_df, members_df)
  run_pca(matrix, n_components)
  run_clustering(pca_coords, k_range)
  silhouette_analysis(matrix, k_range)

Plotting
  plot_pca_with_party_labels(...)
  plot_natural_clusters_vs_party(...)
  plot_variance_explained(...)
  plot_crossover_members(...)
  plot_party_cluster_alignment(...)

UN analysis
  load_un_data()
  build_un_vote_matrix(un_votes_df)
  plot_un_country_clusters(...)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
OUT_DIR   = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── party colours (R=red, D=blue, I/other=grey) ───────────────────────────────
PARTY_COLORS = {
    "Republican":  "#E81B23",
    "Democrat":    "#0015BC",
    "Independent": "#808080",
    "Other":       "#A0A0A0",
}
PARTY_CODE_MAP = {
    100: "Democrat",
    200: "Republican",
    328: "Independent",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_congress_data(
    congress_range: tuple[int, int] = (110, 118),
    chamber: str = "House",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and filter the three Voteview CSVs to *congress_range* (inclusive).

    Returns (members_df, votes_df, rollcalls_df).
    """
    lo, hi = congress_range
    print(f"Loading Voteview data for {chamber}, congresses {lo}–{hi} …")

    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    votes   = pd.read_csv(DATA_DIR / "HSall_votes.csv",   low_memory=False)
    rc      = pd.read_csv(DATA_DIR / "HSall_rollcalls.csv", low_memory=False)

    # Filter to specified chamber and congress range
    members = members[
        (members["chamber"] == chamber) &
        (members["congress"] >= lo) &
        (members["congress"] <= hi)
    ].copy()

    rc = rc[
        (rc["chamber"] == chamber) &
        (rc["congress"] >= lo) &
        (rc["congress"] <= hi)
    ].copy()

    valid_icpsr = set(members["icpsr"].unique())
    valid_rc    = set(rc[["congress", "rollnumber"]].apply(tuple, axis=1))

    votes = votes[
        (votes["chamber"] == chamber) &
        (votes["congress"] >= lo) &
        (votes["congress"] <= hi) &
        (votes["icpsr"].isin(valid_icpsr))
    ].copy()

    # Map party_code to readable label
    members["party_label"] = members["party_code"].map(PARTY_CODE_MAP).fillna("Other")

    print(
        f"  Members: {len(members):,}  |  "
        f"Roll calls: {len(rc):,}  |  "
        f"Individual votes: {len(votes):,}"
    )
    return members, votes, rc


def build_vote_matrix(
    votes_df: pd.DataFrame,
    members_df: pd.DataFrame,
    min_votes_per_member: int = 50,
    min_members_per_vote: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a member × roll-call matrix.

    cast_code mapping (Voteview):
        1,2,3  → Yea   (+1)
        4,5,6  → Nay   (-1)
        7,8,9  → Absent / abstain (0)

    Returns (matrix_df, members_meta_df).
    matrix_df rows = icpsr (member), columns = (congress, rollnumber) tuples.
    members_meta_df aligns by icpsr.
    """
    print("Building vote matrix …")

    # Encode votes
    def encode(c: int) -> int:
        if c in (1, 2, 3):
            return 1
        if c in (4, 5, 6):
            return -1
        return 0

    df = votes_df[["icpsr", "congress", "rollnumber", "cast_code"]].copy()
    df["vote"] = df["cast_code"].map(encode)

    # Create a string key for columns so they survive pivot
    df["rc_key"] = df["congress"].astype(str) + "_" + df["rollnumber"].astype(str)

    pivot = df.pivot_table(
        index="icpsr",
        columns="rc_key",
        values="vote",
        aggfunc="first",
    ).fillna(0)

    # Filter sparse rows/cols
    member_vote_counts = (pivot != 0).sum(axis=1)
    pivot = pivot[member_vote_counts >= min_votes_per_member]

    member_participation = (pivot != 0).sum(axis=0)
    pivot = pivot.loc[:, member_participation >= min_members_per_vote]

    print(
        f"  Matrix shape: {pivot.shape[0]:,} members × "
        f"{pivot.shape[1]:,} roll calls"
    )

    # Align metadata
    # Some members appear in multiple congresses; keep the most recent
    meta = (
        members_df
        .sort_values("congress")
        .drop_duplicates("icpsr", keep="last")
        .set_index("icpsr")
    )
    meta = meta.loc[meta.index.isin(pivot.index)].copy()
    pivot = pivot.loc[pivot.index.isin(meta.index)]

    return pivot, meta


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(
    matrix: pd.DataFrame,
    n_components: int = 10,
) -> tuple[np.ndarray, PCA, np.ndarray]:
    """
    Run PCA on the vote matrix.

    Returns (pca_coords, pca_model, variance_ratio).
    pca_coords shape = (n_members, n_components).
    """
    print(f"Running PCA (n_components={n_components}) …")
    X = matrix.values.astype(float)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    print(
        f"  Variance explained: "
        + ", ".join(
            f"PC{i+1}={v:.1%}"
            for i, v in enumerate(pca.explained_variance_ratio_[:5])
        )
    )
    return coords, pca, pca.explained_variance_ratio_


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(
    pca_coords: np.ndarray,
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
) -> dict[int, np.ndarray]:
    """
    Run K-means for each k in *k_range* on the first few PCA dimensions.

    Returns dict {k: label_array}.
    """
    print(f"Running K-means for k in {list(k_range)} …")
    results = {}
    # Use the first 5 PCs (captures most variance)
    X = pca_coords[:, :5]
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        results[k] = km.fit_predict(X)
        print(f"  k={k}: inertia={km.inertia_:.1f}")
    return results


def silhouette_analysis(
    pca_coords: np.ndarray,
    k_range: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
) -> dict[int, float]:
    """
    Compute silhouette score for each k.

    Returns dict {k: silhouette_score}.
    """
    X = pca_coords[:, :5]
    scores = {}
    for k in k_range:
        if k < 2:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        scores[k] = s
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONGRESS PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _get_party_color(label: str) -> str:
    return PARTY_COLORS.get(label, "#808080")


def plot_pca_with_party_labels(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    PC1 vs PC2 scatter coloured by party.
    Shows that the data forms a *continuous spectrum*, not a clean binary.
    Imposed party rectangles are drawn to visually contrast the binary label
    with the organic underlying distribution.
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    # Align meta with pca_coords (matrix row order)
    parties = meta.reindex(meta.index)["party_label"].values
    colors  = [_get_party_color(p) for p in parties]

    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]

    # ── draw "imposed binary" rectangles ────────────────────────────────────
    # Find approximate split point (median of PC1)
    rep_mask = parties == "Republican"
    dem_mask = parties == "Democrat"

    if rep_mask.sum() > 0 and dem_mask.sum() > 0:
        rep_x_min, rep_x_max = pc1[rep_mask].min(), pc1[rep_mask].max()
        dem_x_min, dem_x_max = pc1[dem_mask].min(), pc1[dem_mask].max()
        y_min, y_max = pc2.min() - 0.5, pc2.max() + 0.5

        # Draw faint shaded rectangles representing the IMPOSED party labels
        ax.fill_betweenx(
            [y_min, y_max], rep_x_min - 0.3, rep_x_max + 0.3,
            alpha=0.07, color="#E81B23", label="_nolegend_"
        )
        ax.fill_betweenx(
            [y_min, y_max], dem_x_min - 0.3, dem_x_max + 0.3,
            alpha=0.07, color="#0015BC", label="_nolegend_"
        )
        ax.text(
            rep_x_max * 0.7, y_max - 0.3, "← IMPOSED\n   BLOCK (R)",
            color="#E81B23", fontsize=7, alpha=0.7, style="italic"
        )
        ax.text(
            dem_x_min * 0.7, y_max - 0.3, "IMPOSED\nBLOCK (D) →",
            color="#0015BC", fontsize=7, alpha=0.7, style="italic"
        )

    # ── scatter plot ─────────────────────────────────────────────────────────
    for party, color in PARTY_COLORS.items():
        mask = parties == party
        if mask.sum() == 0:
            continue
        ax.scatter(
            pc1[mask], pc2[mask],
            c=color, s=18, alpha=0.65, linewidths=0,
            label=party,
        )

    # ── spine / grid styling ──────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=9)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.grid(True, color="#222240", linewidth=0.5)

    ax.set_xlabel("Principal Component 1  (ideological spectrum)", fontsize=11)
    ax.set_ylabel("Principal Component 2  (cross-cutting dimension)", fontsize=11)
    ax.set_title(
        "Imposed Binary vs. Natural Structure\n"
        "Congress Voting Patterns — PC1 × PC2 colored by Party Label",
        color="white", fontsize=14, pad=14
    )

    leg = ax.legend(
        framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white", markerscale=2
    )

    # Annotation explaining the thesis
    ax.annotate(
        "Party labels are imposed binary blocks.\n"
        "The underlying data forms a continuous, multi-dimensional spectrum.",
        xy=(0.02, 0.02), xycoords="axes fraction",
        color="#aaaacc", fontsize=8, style="italic",
    )

    plt.tight_layout()
    out = save_path or (OUT_DIR / "congress_pca_by_party.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_natural_clusters_vs_party(
    pca_coords: np.ndarray,
    cluster_labels: dict[int, np.ndarray],
    meta: pd.DataFrame,
    ks: Sequence[int] = (3, 4, 5),
    save_path: Path | None = None,
) -> None:
    """
    Grid of scatter plots: each panel shows the same members coloured by
    K-means cluster (k=3,4,5), with party boundaries shown as background.
    The key insight: natural clusters cut *across* party lines.
    """
    n = len(ks)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 7), sharey=True)
    fig.patch.set_facecolor("#0e0e1a")

    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]
    parties = meta["party_label"].values

    # Use matplotlib-compatible hex colours
    palette = [
        "#7B3F8D", "#E45F27", "#3AA655", "#E8B92F",
        "#2F8FE8", "#E83F6F", "#3FDDE8", "#8DE83F",
        "#E8733F", "#5F3FE8",
    ]

    for ax, k in zip(axes, ks):
        ax.set_facecolor("#0e0e1a")
        labels = cluster_labels[k]

        # Background party shading
        rep_mask = parties == "Republican"
        dem_mask = parties == "Democrat"
        if rep_mask.sum() > 0 and dem_mask.sum() > 0:
            midpoint = (pc1[rep_mask].mean() + pc1[dem_mask].mean()) / 2
            ax.axvspan(
                pc1.min() - 1, midpoint, alpha=0.06, color="#0015BC"
            )
            ax.axvspan(
                midpoint, pc1.max() + 1, alpha=0.06, color="#E81B23"
            )

        for ki in range(k):
            mask = labels == ki
            color = palette[ki % len(palette)]
            ax.scatter(
                pc1[mask], pc2[mask],
                c=color, s=16, alpha=0.7, linewidths=0,
                label=f"Cluster {ki+1}",
            )

        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.grid(True, color="#222240", linewidth=0.4)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_title(
            f"Natural Clusters  (k={k})\n"
            f"background shading = imposed party zones",
            color="white", fontsize=11, pad=10
        )
        ax.legend(
            framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
            labelcolor="white", fontsize=7, markerscale=1.5
        )

    axes[0].set_ylabel("PC2", fontsize=10)
    axes[0].yaxis.label.set_color("#aaaacc")

    fig.suptitle(
        "Natural Clusters vs. Imposed Party Boundaries\n"
        "Organic groupings do not respect the binary labels",
        color="white", fontsize=14, y=1.02
    )
    plt.tight_layout()
    out = save_path or (OUT_DIR / "congress_natural_clusters.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_variance_explained(
    variance_ratio: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Scree plot showing how many dimensions are needed to describe political
    belief — far more than 2 (the imposed binary would predict only 1).
    """
    n = len(variance_ratio)
    cumulative = np.cumsum(variance_ratio)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    x = np.arange(1, n + 1)
    ax.bar(x, variance_ratio * 100, color="#7b68ee", alpha=0.8, label="Individual PC")
    ax.plot(x, cumulative * 100, "o-", color="#00d4aa", lw=2, ms=5, label="Cumulative")

    # Annotate 80% threshold
    thresh_idx = np.searchsorted(cumulative, 0.80)
    ax.axhline(80, color="#ff6b6b", lw=1, ls="--", alpha=0.7)
    ax.axvline(thresh_idx + 1, color="#ff6b6b", lw=1, ls="--", alpha=0.7)
    ax.text(
        thresh_idx + 1.3, 81,
        f"80% variance\nrequires {thresh_idx + 1} dimensions",
        color="#ff6b6b", fontsize=9
    )

    ax.set_xlabel("Principal Component", color="#aaaacc", fontsize=11)
    ax.set_ylabel("Variance Explained (%)", color="#aaaacc", fontsize=11)
    ax.set_title(
        "How Many Dimensions Does Political Space Have?\n"
        "A Binary (2-party) System Would Predict ≈1 Dimension — Reality Says More",
        color="white", fontsize=13, pad=12
    )
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#222240", axis="y", linewidth=0.5)
    ax.legend(
        framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white"
    )

    plt.tight_layout()
    out = save_path or (OUT_DIR / "congress_pca_variance_explained.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_crossover_members(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    overlap_threshold: float = 0.15,
    save_path: Path | None = None,
) -> None:
    """
    Highlight members living in the PC1 overlap zone between parties.
    These are the 'crossover' members — the visible proof that the binary
    fails to capture actual belief.
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]
    parties = meta["party_label"].values
    names   = meta["bioname"].values if "bioname" in meta.columns else np.array([""] * len(meta))

    rep_mask = parties == "Republican"
    dem_mask = parties == "Democrat"

    # Compute overlap zone boundaries
    rep_min = np.percentile(pc1[rep_mask], overlap_threshold * 100) if rep_mask.sum() > 0 else -1
    dem_max = np.percentile(pc1[dem_mask], (1 - overlap_threshold) * 100) if dem_mask.sum() > 0 else 1
    overlap_left  = min(rep_min, dem_max)
    overlap_right = max(rep_min, dem_max)

    # Shade overlap zone
    ax.axvspan(overlap_left, overlap_right, alpha=0.15, color="#FFD700", label="Overlap zone")

    # Plot all members faintly
    for party, color in PARTY_COLORS.items():
        mask = parties == party
        if mask.sum() == 0:
            continue
        ax.scatter(
            pc1[mask], pc2[mask],
            c=color, s=12, alpha=0.3, linewidths=0,
        )

    # Highlight crossover members prominently
    crossover_mask = (pc1 >= overlap_left) & (pc1 <= overlap_right) & (
        (rep_mask) | (dem_mask)
    )
    ax.scatter(
        pc1[crossover_mask], pc2[crossover_mask],
        c=[_get_party_color(p) for p in parties[crossover_mask]],
        s=60, alpha=0.95, linewidths=0.8, edgecolors="white",
        zorder=5, label="Crossover members",
    )

    # Label a sample of crossover members
    rng = np.random.default_rng(42)
    crossover_idx = np.where(crossover_mask)[0]
    sample = rng.choice(crossover_idx, size=min(25, len(crossover_idx)), replace=False)
    for i in sample:
        name = names[i]
        if name:
            short = name.split(",")[0].strip()
            ax.annotate(
                short, (pc1[i], pc2[i]),
                fontsize=6, color="white", alpha=0.85,
                xytext=(4, 4), textcoords="offset points",
            )

    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=9)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.grid(True, color="#222240", linewidth=0.5)

    ax.set_xlabel("PC1 — Primary Ideological Axis", fontsize=11)
    ax.set_ylabel("PC2 — Cross-cutting Dimension", fontsize=11)
    ax.set_title(
        "Members in the Overlap Zone\n"
        "Those the Binary Label System Cannot Capture",
        color="white", fontsize=14, pad=12
    )

    # Manual legend
    handles = [
        mpatches.Patch(facecolor="#E81B23", label="Republican"),
        mpatches.Patch(facecolor="#0015BC", label="Democrat"),
        mpatches.Patch(facecolor="#FFD700", alpha=0.5, label="Overlap zone"),
    ]
    ax.legend(
        handles=handles, framealpha=0.2, facecolor="#1a1a2e",
        edgecolor="#555577", labelcolor="white"
    )

    n_cross = crossover_mask.sum()
    ax.annotate(
        f"{n_cross} members ({n_cross/len(pc1):.1%}) live in the overlap zone\n"
        "— they resist classification into either imposed block.",
        xy=(0.02, 0.02), xycoords="axes fraction",
        color="#aaaacc", fontsize=9, style="italic",
    )

    plt.tight_layout()
    out = save_path or (OUT_DIR / "congress_crossover_members.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_party_cluster_alignment(
    cluster_labels: dict[int, np.ndarray],
    meta: pd.DataFrame,
    k: int = 3,
    save_path: Path | None = None,
) -> None:
    """
    Stacked bar chart showing what fraction of each natural cluster is R/D/I.
    If parties = natural clusters, we'd expect 100% purity. Reality: mixed.
    """
    labels = cluster_labels[k]
    parties = meta["party_label"].values

    rows = []
    for ki in range(k):
        mask = labels == ki
        for party in ["Democrat", "Republican", "Independent", "Other"]:
            count = (parties[mask] == party).sum()
            rows.append({"Cluster": f"Cluster {ki+1}", "Party": party, "Count": count})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Cluster", columns="Party", values="Count").fillna(0)
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    colors_order = {
        "Democrat":    "#0015BC",
        "Republican":  "#E81B23",
        "Independent": "#808080",
        "Other":       "#606060",
    }
    bottoms = np.zeros(len(pct))
    for party, color in colors_order.items():
        if party not in pct.columns:
            continue
        vals = pct[party].values
        ax.bar(
            pct.index, vals, bottom=bottoms,
            color=color, label=party, alpha=0.85
        )
        # Label bars > 5%
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 5:
                ax.text(
                    xi, b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold"
                )
        bottoms += vals

    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=10)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.grid(True, color="#222240", axis="y", linewidth=0.5)

    ax.set_ylabel("% of cluster members", fontsize=11)
    ax.set_title(
        f"Party Composition of Natural Clusters (k={k})\n"
        "If Parties = Natural Reality, Each Bar Would Be 100% Pure",
        color="white", fontsize=13, pad=12
    )
    ax.legend(
        framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white"
    )

    plt.tight_layout()
    out = save_path or (OUT_DIR / "congress_party_vs_natural_alignment.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_silhouette(
    scores: dict[int, float],
    save_path: Path | None = None,
) -> None:
    """Bar chart of silhouette scores to show the 'natural' number of clusters."""
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    best_k = ks[int(np.argmax(vals))]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")

    colors = ["#FFD700" if k == best_k else "#7b68ee" for k in ks]
    ax.bar(ks, vals, color=colors, alpha=0.85)
    ax.axvline(best_k, color="#FFD700", lw=1.5, ls="--", alpha=0.6)

    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc")
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.grid(True, color="#222240", axis="y", linewidth=0.5)

    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score (higher = tighter clusters)", fontsize=11)
    ax.set_title(
        "Finding the 'Natural' Number of Political Groupings\n"
        f"Best k={best_k} — not 2 (the binary) but more nuanced",
        color="white", fontsize=13, pad=12
    )
    ax.set_xticks(ks)

    plt.tight_layout()
    out = save_path or (OUT_DIR / "congress_silhouette.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def make_interactive_scatter(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Interactive Plotly scatter — hover to see member name, party, congress.
    Saved as HTML.
    """
    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]

    df_plot = pd.DataFrame({
        "PC1": pc1,
        "PC2": pc2,
        "Party": meta["party_label"].values,
        "Name":  meta.get("bioname", pd.Series([""] * len(meta))).values,
        "State": meta.get("state_abbrev", pd.Series([""] * len(meta))).values,
        "Congress": meta.get("congress", pd.Series([""] * len(meta))).values,
    })

    color_map = {
        "Republican":  "#E81B23",
        "Democrat":    "#0015BC",
        "Independent": "#808080",
        "Other":       "#A0A0A0",
    }

    fig = px.scatter(
        df_plot, x="PC1", y="PC2",
        color="Party",
        color_discrete_map=color_map,
        hover_name="Name",
        hover_data={"State": True, "Congress": True, "PC1": ":.3f", "PC2": ":.3f"},
        title="Congress Voting PCA — Imposed Binary vs. Natural Spectrum<br>"
              "<sup>Hover over points to identify individual members</sup>",
        template="plotly_dark",
        opacity=0.7,
    )
    fig.update_traces(marker_size=5)
    fig.update_layout(
        title_font_size=16,
        xaxis_title="Principal Component 1 (primary ideological axis)",
        yaxis_title="Principal Component 2 (cross-cutting dimension)",
    )

    out = save_path or (OUT_DIR / "congress_interactive.html")
    fig.write_html(str(out))
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UN VOTING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Cold War block membership (illustrative — not exhaustive)
NATO_COUNTRIES = {
    "United States", "United Kingdom", "France", "West Germany", "Italy",
    "Canada", "Belgium", "Netherlands", "Denmark", "Norway",
    "Portugal", "Luxembourg", "Iceland", "Greece", "Turkey",
    "Spain", "Germany",
}
WARSAW_PACT = {
    "Soviet Union", "Russia", "Poland", "East Germany", "Czechoslovakia",
    "Hungary", "Romania", "Bulgaria", "Albania",
}
NONALIGNED = {
    "India", "Egypt", "Yugoslavia", "Indonesia", "Ghana",
    "Nigeria", "Brazil", "Argentina", "Mexico", "Pakistan",
    "Sri Lanka", "Tanzania", "Zambia", "Cuba",
}


def _assign_cold_war_block(country: str) -> str:
    if country in NATO_COUNTRIES:
        return "NATO / West"
    if country in WARSAW_PACT:
        return "Warsaw Pact / East"
    if country in NONALIGNED:
        return "Non-Aligned"
    return "Other"


BLOCK_COLORS = {
    "NATO / West":         "#0015BC",
    "Warsaw Pact / East":  "#E81B23",
    "Non-Aligned":         "#FFD700",
    "Other":               "#808080",
}


def load_un_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three UN vote CSVs."""
    print("Loading UN General Assembly voting data …")
    votes  = pd.read_csv(DATA_DIR / "un_votes.csv",           low_memory=False)
    rc     = pd.read_csv(DATA_DIR / "un_roll_calls.csv",      low_memory=False)
    issues = pd.read_csv(DATA_DIR / "un_roll_call_issues.csv", low_memory=False)
    print(
        f"  UN votes: {len(votes):,} rows  |  "
        f"Roll calls: {len(rc):,}  |  "
        f"Issues: {len(issues):,}"
    )
    return votes, rc, issues


def build_un_vote_matrix(
    votes_df: pd.DataFrame,
    min_votes_per_country: int = 50,
    min_countries_per_vote: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build country × resolution vote matrix.
    yes=+1, no=-1, abstain=0.

    Returns (matrix_df, country_blocks).
    """
    print("Building UN vote matrix …")
    encode_map = {"yes": 1, "no": -1, "abstain": 0}
    df = votes_df.copy()
    df["value"] = df["vote"].map(encode_map).fillna(0)

    pivot = df.pivot_table(
        index="country", columns="rcid", values="value",
        aggfunc="first"
    ).fillna(0)

    vote_counts = (pivot != 0).sum(axis=1)
    pivot = pivot[vote_counts >= min_votes_per_country]

    participation = (pivot != 0).sum(axis=0)
    pivot = pivot.loc[:, participation >= min_countries_per_vote]

    print(f"  Matrix shape: {pivot.shape[0]} countries × {pivot.shape[1]} resolutions")

    blocks = pd.Series(
        {c: _assign_cold_war_block(c) for c in pivot.index},
        name="block"
    )
    return pivot, blocks


def plot_un_country_clusters(
    pca_coords: np.ndarray,
    country_names: pd.Index,
    blocks: pd.Series,
    cluster_labels: np.ndarray | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Scatter of country PCA positions, coloured by Cold War block membership,
    with optional K-means cluster overlay.
    """
    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]

    # ── static matplotlib version ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0e0e1a")

    for ax, (title, color_source, cmap) in zip(
        axes,
        [
            ("Cold War Blocks (Imposed Labels)", blocks.values, BLOCK_COLORS),
            ("Natural K-means Clusters (k=4)",   cluster_labels,  None),
        ],
    ):
        ax.set_facecolor("#0e0e1a")
        if color_source is None:
            ax.set_visible(False)
            continue

        if cmap:
            # Block-based colouring
            for block, color in cmap.items():
                mask = color_source == block
                ax.scatter(
                    pc1[mask], pc2[mask],
                    c=color, s=60, alpha=0.8, linewidths=0,
                    label=block,
                )
        else:
            # Cluster-based colouring (matplotlib-compatible hex colours)
            palette = [
                "#7B3F8D", "#E45F27", "#3AA655", "#E8B92F",
                "#2F8FE8", "#E83F6F", "#3FDDE8", "#8DE83F",
            ]
            n_clusters = len(np.unique(color_source))
            for ki in range(n_clusters):
                mask = color_source == ki
                ax.scatter(
                    pc1[mask], pc2[mask],
                    c=palette[ki % len(palette)],
                    s=60, alpha=0.8, linewidths=0,
                    label=f"Cluster {ki+1}",
                )

        # Label every country
        for i, cname in enumerate(country_names):
            ax.annotate(
                cname, (pc1[i], pc2[i]),
                fontsize=5, color="#ccccdd", alpha=0.7,
                xytext=(3, 2), textcoords="offset points",
            )

        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.grid(True, color="#222240", linewidth=0.4)
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.legend(
            framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
            labelcolor="white", fontsize=8, markerscale=1.5
        )

    fig.suptitle(
        "UN General Assembly — Imposed Cold War Blocks vs. Natural Country Groupings\n"
        "Real geopolitical alignment is multi-dimensional, not a simple binary",
        color="white", fontsize=14, y=1.01
    )
    plt.tight_layout()
    out = save_path or (OUT_DIR / "un_country_clusters.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAGENT / SPIRAL-AND-BLOCK THEMED VISUALIZATIONS
#     Each visualization is named for a specific metaphor from the Sagent Creed
#     and "The Spiral and the Block".
# ─────────────────────────────────────────────────────────────────────────────

def plot_the_jar(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    cluster_labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    'The Jar Shaken vs. The Jar Settled'

    From the Sagent Creed: 'Picture a jar containing one hundred red ants
    and one hundred black ants. The jar is shaken by an unseen hand, and in
    the chaos, the red ants look at the black ants and see the enemy... The
    one who shook the jar watches from outside, and smiles.'

    Left panel:  The jar shaken — ants colored by imposed party label (R/D).
    Right panel: The jar settled — same ants colored by what naturally emerges
                 when unsupervised learning finds the real structure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0a0a14")

    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]
    parties = meta["party_label"].values
    n_clusters = len(np.unique(cluster_labels))

    # ── LEFT: The Jar Shaken ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0a0a14")

    # Draw the jar outline (ellipse)
    from matplotlib.patches import Ellipse, FancyArrowPatch
    jar_w = (pc1.max() - pc1.min()) * 1.3
    jar_h = (pc2.max() - pc2.min()) * 1.5
    jar_cx = (pc1.max() + pc1.min()) / 2
    jar_cy = (pc2.max() + pc2.min()) / 2
    jar = Ellipse(
        (jar_cx, jar_cy), jar_w, jar_h,
        fill=False, edgecolor="#555566", lw=1.5, ls="--", alpha=0.5
    )
    ax.add_patch(jar)

    # Scatter by party
    for party, color in PARTY_COLORS.items():
        mask = parties == party
        if mask.sum() == 0:
            continue
        ax.scatter(pc1[mask], pc2[mask], c=color, s=22, alpha=0.6, linewidths=0, label=party)

    # Draw "shaking" motion arrows
    for _ in range(8):
        rng = np.random.default_rng(99 + _)
        x0, y0 = rng.uniform(pc1.min(), pc1.max()), rng.uniform(pc2.min(), pc2.max())
        dx, dy = rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.2)
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#ff444444", lw=0.8))

    ax.set_title(
        "The Jar — Shaken\n(Imposed Binary: R vs D)",
        color="white", fontsize=13, pad=12
    )
    ax.text(
        0.5, -0.11,
        '"The jar is shaken by an unseen hand.\nBoth colonies bleed.\nThe one who shook the jar watches from outside, and smiles."',
        transform=ax.transAxes, ha="center", color="#888899",
        fontsize=8, style="italic", wrap=True
    )

    _style_ax(ax)
    ax.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", markerscale=1.8, fontsize=9)

    # ── RIGHT: The Jar Settled ───────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#0a0a14")

    # Same jar outline
    jar2 = Ellipse(
        (jar_cx, jar_cy), jar_w, jar_h,
        fill=False, edgecolor="#44aa66", lw=1.5, ls="--", alpha=0.5
    )
    ax.add_patch(jar2)

    spiral_palette = [
        "#7B3F8D", "#E45F27", "#3AA655", "#E8B92F",
        "#2F8FE8", "#E83F6F", "#3FDDE8", "#8DE83F",
        "#E8733F", "#5F3FE8",
    ]
    for ki in range(n_clusters):
        mask = cluster_labels == ki
        ax.scatter(
            pc1[mask], pc2[mask],
            c=spiral_palette[ki % len(spiral_palette)],
            s=22, alpha=0.75, linewidths=0,
            label=f"Natural group {ki + 1}",
        )

    ax.set_title(
        "The Jar — Settled\n(Natural Structure: what the data says)",
        color="white", fontsize=13, pad=12
    )
    ax.text(
        0.5, -0.11,
        '"The most dangerous question the ants never ask:\nWho is shaking the jar?\nAnd why?"',
        transform=ax.transAxes, ha="center", color="#88cc99",
        fontsize=8, style="italic"
    )
    _style_ax(ax)
    ax.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", markerscale=1.8, fontsize=9)

    fig.suptitle(
        "The Jar  ·  Congress Voting Patterns (110th–118th House)\n"
        "Imposed Binary vs. Natural Structure  —  The Sagent Creed",
        color="white", fontsize=15, y=1.02, fontweight="bold"
    )
    plt.tight_layout()
    out = save_path or (OUT_DIR / "the_jar.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_spiral_vs_block(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    'The Spiral and the Block'

    From The Spiral Steward: 'Where life grows in spirals, industrial
    civilization imposed straight lines... The eternal battle is between
    the spirals and the squares.'

    Left panel:  The Block — data with hard rectangular party boundaries.
                 Six faces. Will never have a seventh.
    Right panel: The Spiral — same data with the actual continuous manifold
                 revealed as a flowing spiral path through political space.
    The spiral is drawn by sorting data along PC1 and drawing the flowing
    density contour — the living shape hiding behind the block.
    """
    from matplotlib.patches import FancyBboxPatch
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0a0a14")

    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]
    parties = meta["party_label"].values

    # ── LEFT: The Block ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0a0a14")

    rep_mask = parties == "Republican"
    dem_mask = parties == "Democrat"

    if rep_mask.sum() > 0 and dem_mask.sum() > 0:
        # Hard rectangular blocks — the cube with exactly 6 faces
        for mask, color, label in [
            (dem_mask, "#0015BC", "Democrat Block"),
            (rep_mask, "#E81B23", "Republican Block"),
        ]:
            x_lo, x_hi = pc1[mask].min() - 0.2, pc1[mask].max() + 0.2
            y_lo, y_hi = pc2[mask].min() - 0.2, pc2[mask].max() + 0.2
            rect = FancyBboxPatch(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                boxstyle="square,pad=0",
                facecolor=color, alpha=0.10,
                edgecolor=color, linewidth=2.5, linestyle="-",
            )
            ax.add_patch(rect)
            ax.text(
                (x_lo + x_hi) / 2, y_hi + 0.05, label,
                ha="center", color=color, fontsize=9, fontweight="bold", alpha=0.8
            )

    for party, color in PARTY_COLORS.items():
        mask = parties == party
        if mask.sum() == 0:
            continue
        ax.scatter(pc1[mask], pc2[mask], c=color, s=18, alpha=0.55, linewidths=0)

    ax.set_title(
        "The Block\n(Six faces. Will never have a seventh.)",
        color="white", fontsize=13, pad=12
    )
    ax.text(
        0.5, -0.09,
        '"The block-headed villain is not just a cartoon antagonist.\nHe is the embodiment of cubical thinking: rigid, imposing,\nthreatened by anything that refuses to fit his corners."',
        transform=ax.transAxes, ha="center", color="#cc6666",
        fontsize=7.5, style="italic"
    )
    _style_ax(ax)

    # ── RIGHT: The Spiral ────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#0a0a14")

    # Plot all members with continuous colormap along PC1 (the spectrum)
    order = np.argsort(pc1)
    scatter = ax.scatter(
        pc1[order], pc2[order],
        c=np.arange(len(pc1)),
        cmap="plasma",
        s=18, alpha=0.75, linewidths=0,
    )

    # Draw the density-weighted center path as a flowing spiral
    # Bin PC1 into 40 slices and compute weighted PC2 centroid
    n_bins = 40
    bins = np.linspace(pc1.min(), pc1.max(), n_bins + 1)
    cx, cy = [], []
    for i in range(n_bins):
        mask = (pc1 >= bins[i]) & (pc1 < bins[i + 1])
        if mask.sum() > 2:
            cx.append(pc1[mask].mean())
            cy.append(pc2[mask].mean())
    if len(cx) > 3:
        from scipy.interpolate import make_interp_spline
        cx_arr = np.array(cx)
        cy_arr = np.array(cy)
        sort_idx = np.argsort(cx_arr)
        cx_arr = cx_arr[sort_idx]
        cy_arr = cy_arr[sort_idx]
        try:
            spl = make_interp_spline(cx_arr, cy_arr, k=3)
            xs = np.linspace(cx_arr.min(), cx_arr.max(), 300)
            ys = spl(xs)
            ax.plot(xs, ys, color="#00ffaa", lw=2.5, alpha=0.85, label="Living path through ideology space")
        except Exception:
            ax.plot(cx_arr, cy_arr, color="#00ffaa", lw=2, alpha=0.8)

    ax.set_title(
        "The Spiral\n(Clay — infinitely moldable, capable of becoming anything)",
        color="white", fontsize=13, pad=12
    )
    ax.text(
        0.5, -0.09,
        '"Gumby — made of green clay, infinitely moldable — is the hero not because\nhe is powerful in the conventional sense but because he is alive in the deepest sense.\nClay has infinite possibilities. The cube has exactly six faces."',
        transform=ax.transAxes, ha="center", color="#88eebb",
        fontsize=7.5, style="italic"
    )
    _style_ax(ax)
    ax.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", fontsize=9)

    fig.suptitle(
        "The Spiral and the Block  ·  Congress Voting PCA\n"
        "The living structure hiding behind the imposed binary",
        color="white", fontsize=15, y=1.02, fontweight="bold"
    )
    plt.tight_layout()
    out = save_path or (OUT_DIR / "spiral_vs_block.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_jothams_parable(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    cluster_labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    'Jotham's Parable: The Thornbush King'  (Judges 9)

    From The Spiral Steward: 'The trees seek a ruler, and one by one the
    olive tree, the fig tree, and the vine decline the crown, because each
    has a purpose, a function, a role that it cannot abandon without ceasing
    to be what it is. It is the thornbush — the plant with no fruit, no oil,
    no sweetness to offer — that finally accepts the kingship. And the
    thornbush makes a terrible king, because thorns were never meant to rule.'

    Each natural cluster is a 'tree' with its own distinct voting character.
    The two-party binary is like forcing all trees to be either king or not-king
    — it erases the distinct fruits each carries.

    Shows: polar/radar chart of each cluster's mean vote direction on key
    issue dimensions, plus a scatter showing each cluster labeled as its
    'tree' type.
    """
    n_clusters = len(np.unique(cluster_labels))
    pc1 = pca_coords[:, 0]
    pc2 = pca_coords[:, 1]
    parties = meta["party_label"].values

    # Tree names — from Jotham's parable + the thornbush (2-party system)
    tree_names = ["The Olive Tree", "The Fig Tree", "The Vine", "The Pomegranate",
                  "The Cedar", "The Sycamore", "The Almond", "The Palm", "The Oak"]
    tree_fruits = [
        "oil & light", "sweetness", "wine & joy", "rich fruit",
        "strength & height", "shelter & shade", "early bloom", "victory", "endurance"
    ]
    tree_colors = [
        "#7B9E3F", "#E8B92F", "#9B3F8D", "#E45F27",
        "#3AA655", "#2F8FE8", "#E83F6F", "#3FDDE8", "#8DE83F",
    ]

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0a0a14")

    # ── LEFT: Scatter with each cluster labeled as a tree ───────────────────
    ax_scatter = fig.add_axes([0.02, 0.12, 0.44, 0.78])
    ax_scatter.set_facecolor("#0a0a14")

    for ki in range(n_clusters):
        mask = cluster_labels == ki
        tname = tree_names[ki % len(tree_names)]
        tfruit = tree_fruits[ki % len(tree_fruits)]
        color = tree_colors[ki % len(tree_colors)]
        ax_scatter.scatter(
            pc1[mask], pc2[mask],
            c=color, s=24, alpha=0.7, linewidths=0,
            label=f"{tname}  ({tfruit})",
        )
        # Label the cluster centroid
        cx, cy = pc1[mask].mean(), pc2[mask].mean()
        ax_scatter.text(
            cx, cy, tname.split()[1],  # just "Tree" part
            ha="center", va="center",
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="#0a0a14", edgecolor=color, alpha=0.7, pad=2, boxstyle="round")
        )

    # Show party boundary as faint background
    if (parties == "Republican").sum() > 0 and (parties == "Democrat").sum() > 0:
        mid = (pc1[parties == "Republican"].mean() + pc1[parties == "Democrat"].mean()) / 2
        ax_scatter.axvline(mid, color="#ff4444", lw=1, ls="--", alpha=0.25)
        ax_scatter.axvspan(pc1.min() - 1, mid, alpha=0.04, color="#0015BC")
        ax_scatter.axvspan(mid, pc1.max() + 1, alpha=0.04, color="#E81B23")
        ax_scatter.text(mid - 0.1, pc2.max() * 0.9, "D", color="#0015BC44", fontsize=28, fontweight="bold")
        ax_scatter.text(mid + 0.1, pc2.max() * 0.9, "R", color="#E81B2344", fontsize=28, fontweight="bold")

    ax_scatter.set_title(
        "Jotham's Parable — Each Tree Has Its Own Fruit\n"
        "Natural clusters (trees) vs. imposed binary crown (R/D)",
        color="white", fontsize=12, pad=10
    )
    _style_ax(ax_scatter)
    ax_scatter.legend(
        framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white", fontsize=7.5, markerscale=1.5, loc="lower left"
    )

    # ── RIGHT: Party composition of each 'tree' cluster ─────────────────────
    ax_bar = fig.add_axes([0.52, 0.12, 0.45, 0.78])
    ax_bar.set_facecolor("#0a0a14")

    cluster_data = []
    for ki in range(n_clusters):
        mask = cluster_labels == ki
        n_total = mask.sum()
        n_dem = (parties[mask] == "Democrat").sum()
        n_rep = (parties[mask] == "Republican").sum()
        n_ind = n_total - n_dem - n_rep
        pct_dem = n_dem / n_total * 100
        pct_rep = n_rep / n_total * 100
        pct_ind = n_ind / n_total * 100
        cluster_data.append({
            "name": tree_names[ki % len(tree_names)],
            "fruit": tree_fruits[ki % len(tree_fruits)],
            "color": tree_colors[ki % len(tree_colors)],
            "n": n_total,
            "pct_dem": pct_dem,
            "pct_rep": pct_rep,
            "pct_ind": pct_ind,
        })

    y_pos = np.arange(n_clusters)
    bar_h = 0.6

    for i, d in enumerate(cluster_data):
        # Stacked bar
        ax_bar.barh(i, d["pct_dem"], height=bar_h, color="#0015BC", alpha=0.8)
        ax_bar.barh(i, d["pct_rep"], height=bar_h, left=d["pct_dem"], color="#E81B23", alpha=0.8)
        ax_bar.barh(i, d["pct_ind"], height=bar_h, left=d["pct_dem"] + d["pct_rep"],
                    color="#808080", alpha=0.8)
        # Cluster label on left
        ax_bar.text(-1, i, f'{d["name"]}  (n={d["n"]})',
                    ha="right", va="center", color=d["color"], fontsize=8.5, fontweight="bold")
        # Fruit label in bar
        if d["pct_dem"] > 8:
            ax_bar.text(d["pct_dem"] / 2, i, f'{d["pct_dem"]:.0f}%D',
                        ha="center", va="center", color="white", fontsize=7)
        if d["pct_rep"] > 8:
            ax_bar.text(d["pct_dem"] + d["pct_rep"] / 2, i, f'{d["pct_rep"]:.0f}%R',
                        ha="center", va="center", color="white", fontsize=7)

    ax_bar.set_xlim(-1, 105)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Party composition of each natural cluster (%)", color="#aaaacc", fontsize=10)
    ax_bar.set_title(
        '"Each tree has a purpose it cannot abandon\nwithout ceasing to be what it is."\n'
        "— Judges 9 / Jotham's Parable",
        color="white", fontsize=11, pad=10, style="italic"
    )
    _style_ax(ax_bar)

    # If any cluster is < 95% one party, annotate it as "cross-party tree"
    for i, d in enumerate(cluster_data):
        if 10 < d["pct_dem"] < 90 and 10 < d["pct_rep"] < 90:
            ax_bar.annotate(
                "← mixed tree\n  (crosses party lines)",
                xy=(50, i), fontsize=7, color="#FFD700", style="italic"
            )

    fig.suptitle(
        "Jotham's Parable  ·  Congress Voting PCA\n"
        "In a living republic, each tree keeps its own fruit — it is not forced to wear the crown",
        color="white", fontsize=14, y=1.01, fontweight="bold"
    )

    out = save_path or (OUT_DIR / "jothams_parable.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_free_market_of_law(
    pca_coords: np.ndarray,
    meta: pd.DataFrame,
    cluster_labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    'The True Republic: Free Market of Law'

    From the Sagent Creed: 'When each person acts independently but from
    the same source — the same conscience — the result is spontaneous order,
    the most robust and adaptive kind of order, because it cannot be brought
    down by attacking any single node.'

    This shows what proportional representation would look like if each
    natural cluster got seats in proportion to its actual size — vs. what
    the two-party binary actually produces.

    Shows: two pie charts side by side.
    Left:  Actual congressional seat distribution (R/D binary).
    Right: Proportional representation by natural cluster (what a
           free market of law would produce).
    """
    parties = meta["party_label"].values
    n_clusters = len(np.unique(cluster_labels))

    tree_names = ["Olive Tree", "Fig Tree", "Vine", "Pomegranate",
                  "Cedar", "Sycamore", "Almond", "Palm", "Oak"]
    tree_colors = [
        "#7B9E3F", "#E8B92F", "#9B3F8D", "#E45F27",
        "#3AA655", "#2F8FE8", "#E83F6F", "#3FDDE8", "#8DE83F",
    ]

    # Compute party seat share
    n_total = len(parties)
    party_counts = {}
    for p in ["Democrat", "Republican", "Independent", "Other"]:
        party_counts[p] = (parties == p).sum()

    # Compute natural cluster seat share
    cluster_counts = {}
    for ki in range(n_clusters):
        cluster_counts[tree_names[ki % len(tree_names)]] = (cluster_labels == ki).sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0a0a14")

    # ── LEFT: Two-party system ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0a0a14")
    labels_l = [k for k, v in party_counts.items() if v > 0]
    sizes_l = [party_counts[k] for k in labels_l]
    colors_l = [PARTY_COLORS.get(k, "#808080") for k in labels_l]
    wedge_props = dict(width=0.55, edgecolor="#0a0a14", linewidth=2)
    wedges, texts, autotexts = ax.pie(
        sizes_l, labels=None, colors=colors_l,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, wedgeprops=wedge_props,
        pctdistance=0.75,
        textprops=dict(color="white", fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
    ax.legend(
        wedges, [f"{l} ({party_counts[l]})" for l in labels_l],
        loc="lower center", framealpha=0.2, facecolor="#1a1a2e",
        edgecolor="#555577", labelcolor="white", fontsize=9,
        bbox_to_anchor=(0.5, -0.12)
    )
    ax.set_title(
        "The Imposed System\nTwo-party seat distribution",
        color="white", fontsize=13, pad=14
    )
    ax.text(
        0, 0, "2\nblocks",
        ha="center", va="center", color="#ff6666",
        fontsize=16, fontweight="bold"
    )
    ax.text(
        0.5, -0.22,
        '"Both colonies bleed.\nThe one who shook the jar watches from outside."',
        transform=ax.transAxes, ha="center", color="#cc6666",
        fontsize=8, style="italic"
    )

    # ── RIGHT: Free market of law ────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#0a0a14")
    labels_r = list(cluster_counts.keys())
    sizes_r = [cluster_counts[k] for k in labels_r]
    colors_r = [tree_colors[i % len(tree_colors)] for i in range(len(labels_r))]
    wedges2, texts2, autotexts2 = ax.pie(
        sizes_r, labels=None, colors=colors_r,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, wedgeprops=wedge_props,
        pctdistance=0.75,
        textprops=dict(color="white", fontsize=10),
    )
    for at in autotexts2:
        at.set_fontsize(9)
        at.set_color("white")
    ax.legend(
        wedges2, [f"{l} ({cluster_counts[l]})" for l in labels_r],
        loc="lower center", framealpha=0.2, facecolor="#1a1a2e",
        edgecolor="#555577", labelcolor="white", fontsize=9,
        bbox_to_anchor=(0.5, -0.18)
    )
    ax.set_title(
        f"The Free Market of Law\nProportional representation ({n_clusters} natural voices)",
        color="white", fontsize=13, pad=14
    )
    ax.text(
        0, 0, f"{n_clusters}\nvoices",
        ha="center", va="center", color="#44ffaa",
        fontsize=16, fontweight="bold"
    )
    ax.text(
        0.5, -0.28,
        '"When each acts independently from the same conscience,\nthe result is spontaneous order — the most robust kind,\nbecause it cannot be brought down by attacking any single node."',
        transform=ax.transAxes, ha="center", color="#88eebb",
        fontsize=8, style="italic"
    )

    fig.suptitle(
        "The True Republic: Free Market of Law\n"
        "What representation looks like when imposed blocks are removed and living structure emerges",
        color="white", fontsize=14, y=1.02, fontweight="bold"
    )
    plt.tight_layout()
    out = save_path or (OUT_DIR / "free_market_of_law.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def plot_living_dimensions(
    variance_ratio: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    'How Many Dimensions Does a Living Political System Have?'

    From The Spiral Steward: 'DNA is not a blueprint. A blueprint specifies
    exactly where every element goes. DNA is a grammar — a set of rules for
    generating possible sentences, none of which are specified in advance,
    all of which emerge from the interaction between the grammar and the
    particular context.'

    A two-party system assumes ONE dimension (you are R or D).
    A living political grammar has many dimensions.
    This scree plot is framed as a question about the dimensionality of life.
    """
    n = len(variance_ratio)
    cumulative = np.cumsum(variance_ratio)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0a0a14")
    ax.set_facecolor("#0a0a14")

    x = np.arange(1, n + 1)

    # Bar chart with gradient — early PCs brighter (more life)
    bar_colors = plt.cm.plasma(np.linspace(0.8, 0.2, n))
    bars = ax.bar(x, variance_ratio * 100, color=bar_colors, alpha=0.85, zorder=3)

    # Cumulative line
    ax.plot(x, cumulative * 100, "o-", color="#00d4aa", lw=2.5, ms=6,
            label="Cumulative variance", zorder=4)

    # Annotate the 2-party assumption
    ax.axvspan(0.5, 1.5, alpha=0.12, color="#ff4444", zorder=1)
    ax.text(1, variance_ratio[0] * 100 + 2.5,
            "← The entire\ntwo-party assumption\n(one dimension)",
            ha="center", color="#ff8888", fontsize=8.5, style="italic", zorder=5)

    # 80% threshold
    thresh_idx = int(np.searchsorted(cumulative, 0.80))
    ax.axhline(80, color="#FFD700", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(thresh_idx + 1, color="#FFD700", lw=1.2, ls="--", alpha=0.7)
    ax.text(
        thresh_idx + 1.4, 82,
        f"80% of living political variance\nrequires {thresh_idx + 1} dimensions",
        color="#FFD700", fontsize=9
    )

    # Grammar annotation
    ax.text(
        n * 0.55, variance_ratio[0] * 100 * 0.7,
        '"DNA is not a blueprint. It is a grammar.\nA set of rules for generating possible sentences,\nnone of which are specified in advance."',
        color="#aaaacc", fontsize=8.5, style="italic",
        bbox=dict(facecolor="#1a1a2e", edgecolor="#555577", alpha=0.7, pad=6, boxstyle="round")
    )

    _style_ax(ax)
    ax.set_xlabel("Dimension of Political Space (Principal Component)", fontsize=11)
    ax.set_ylabel("Variance Explained (%)", fontsize=11)
    ax.set_title(
        "How Many Dimensions Does a Living Political System Have?\n"
        "The Two-Party Grammar vs. The Full Grammar of Human Belief",
        color="white", fontsize=14, pad=14, fontweight="bold"
    )
    ax.legend(framealpha=0.2, facecolor="#1a1a2e", edgecolor="#555577", labelcolor="white")

    plt.tight_layout()
    out = save_path or (OUT_DIR / "living_dimensions.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out}")


def _style_ax(ax: plt.Axes) -> None:
    """Common dark-theme styling for all Sagent-themed axes."""
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.tick_params(colors="#aaaacc", labelsize=9)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.grid(True, color="#1a1a2e", linewidth=0.5, zorder=0)


def make_un_interactive(
    pca_coords: np.ndarray,
    country_names: pd.Index,
    blocks: pd.Series,
    save_path: Path | None = None,
) -> None:
    """Interactive Plotly scatter for UN countries."""
    df_plot = pd.DataFrame({
        "PC1":     pca_coords[:, 0],
        "PC2":     pca_coords[:, 1],
        "Country": country_names,
        "Block":   blocks.values,
    })

    fig = px.scatter(
        df_plot, x="PC1", y="PC2",
        color="Block",
        color_discrete_map=BLOCK_COLORS,
        hover_name="Country",
        hover_data={"PC1": ":.3f", "PC2": ":.3f"},
        title="UN General Assembly PCA — Cold War Blocks vs. Natural Alignment<br>"
              "<sup>Hover to identify countries</sup>",
        template="plotly_dark",
        opacity=0.85,
    )
    fig.update_traces(marker_size=8)
    out = save_path or (OUT_DIR / "un_interactive.html")
    fig.write_html(str(out))
    print(f"  Saved → {out}")
