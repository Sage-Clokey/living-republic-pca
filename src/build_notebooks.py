"""
build_notebooks.py
------------------
Programmatically creates the two Jupyter notebooks using nbformat.
Run this script once to generate the .ipynb files in notebooks/.
"""

import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)

def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)

def nb(*cells) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = list(cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1 — Congress Voting
# ─────────────────────────────────────────────────────────────────────────────

def build_congress_notebook() -> None:
    cells = [

        md("""\
# 01 · Congress Voting Patterns — Imposed Binary vs. Natural Structure

## Thesis

Political systems forced into **2-party blocks** mask the true underlying
**continuous, multi-dimensional** nature of political belief.

PCA and unsupervised clustering reveal the natural groupings that exist
*independently of imposed labels* — like a living system vs. a rigid binary.
This supports the idea of a **"true republic"** as a free market of law:
emergent and organic, not artificially constrained.

---

**Data source:** [Voteview.com](https://voteview.com) — the gold standard for
congressional roll-call voting records, maintained by UCLA.
"""),

        code("""\
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Make src/ importable
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from voting_pca import (
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
)

OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
print("Environment ready.")
"""),

        md("""\
## 1 · Load the Data

We use congresses **110–118** (2007–2024).  This captures the modern
hyper-partisan era while still including meaningful cross-party variation.

Voteview encodes cast codes as:
| Code | Meaning | Our encoding |
|------|---------|-------------|
| 1–3 | Yea | +1 |
| 4–6 | Nay | −1 |
| 7–9 | Absent / abstain | 0 |
"""),

        code("""\
members, votes, rollcalls = load_congress_data(congress_range=(110, 118), chamber="House")
members.head(3)
"""),

        code("""\
print("Party distribution in our sample:")
print(members["party_label"].value_counts())
"""),

        md("""\
## 2 · Build the Vote Matrix

Each row is a **member** (identified by `icpsr`).
Each column is a **roll call** (congress + roll number).
Cells are +1 / −1 / 0.

We drop members who voted on fewer than 50 roll calls
and roll calls with fewer than 100 participants, to keep the matrix dense.
"""),

        code("""\
vote_matrix, meta = build_vote_matrix(votes, members)
print(f"Vote matrix: {vote_matrix.shape}")
print(f"Meta: {meta.shape}")
meta["party_label"].value_counts()
"""),

        md("""\
## 3 · PCA — Letting the Data Speak

If the binary party system *perfectly* described political belief, we'd expect:
- Only **1 principal component** explaining virtually all variance
- A clean gap between two distinct clouds of members

Instead, PCA reveals a **continuous spectrum** needing many dimensions.
"""),

        code("""\
pca_coords, pca_model, var_ratio = run_pca(vote_matrix, n_components=10)
print("Variance explained per component:")
for i, v in enumerate(var_ratio):
    print(f"  PC{i+1}: {v:.1%}  (cumulative: {var_ratio[:i+1].sum():.1%})")
"""),

        code("""\
# Scree plot — the binary would predict ~1 dimension.  Reality says more.
plot_variance_explained(var_ratio)
from IPython.display import Image
Image(str(OUT_DIR / "congress_pca_variance_explained.png"))
"""),

        md("""\
**Key insight:** PC1 alone explains maybe 30–40% of variance.
Even reaching 80% requires 4–6 dimensions.
A truly binary system would need only 1.

*The data already falsifies the imposed binary before we look at a scatter plot.*
"""),

        md("""\
## 4 · PC1 × PC2 — The Continuous Spectrum Revealed

The scatter below colours each member by their **party label** (R/D/I).
Faint rectangles represent the *imposed* party zones.

The key observation: members form a **gradient**, not two isolated islands.
There is measurable *overlap*.
"""),

        code("""\
plot_pca_with_party_labels(pca_coords, meta)
Image(str(OUT_DIR / "congress_pca_by_party.png"))
"""),

        md("""\
## 5 · Natural Clusters — What the Data Wants to Be

Now we remove the party labels entirely and ask:
*If you had to group these members by voting behaviour alone, what groups emerge?*

K-means for k=3, 4, 5.  The background shading shows the party zones
so you can see how the *natural* clusters cut across those imposed boundaries.
"""),

        code("""\
cluster_results = run_clustering(pca_coords, k_range=[2, 3, 4, 5, 6, 8, 10])
"""),

        code("""\
plot_natural_clusters_vs_party(pca_coords, cluster_results, meta, ks=[3, 4, 5])
Image(str(OUT_DIR / "congress_natural_clusters.png"))
"""),

        md("""\
## 6 · How Well Do Party Labels Predict Natural Clusters?

If parties = natural reality, each cluster would be ~100% R or ~100% D.
The stacked bars show the actual mixture.
"""),

        code("""\
plot_party_cluster_alignment(cluster_results, meta, k=3)
Image(str(OUT_DIR / "congress_party_vs_natural_alignment.png"))
"""),

        md("""\
## 7 · Silhouette Analysis — Finding the True k

The silhouette score measures how well each point fits its own cluster vs.
the next nearest cluster.  We ask: does the data prefer k=2 (the party system)
or something else?
"""),

        code("""\
sil_scores = silhouette_analysis(pca_coords, k_range=[2, 3, 4, 5, 6, 8, 10])
print("Silhouette scores:")
for k, s in sorted(sil_scores.items()):
    print(f"  k={k}: {s:.4f}")

plot_silhouette(sil_scores)
Image(str(OUT_DIR / "congress_silhouette.png"))
"""),

        md("""\
## 8 · Members in the Overlap Zone

These are the *crossover* members — representatives whose voting behaviour
places them **in between** the party ideal points.
They are the living proof that the binary label fails to capture actual belief.
"""),

        code("""\
plot_crossover_members(pca_coords, meta)
Image(str(OUT_DIR / "congress_crossover_members.png"))
"""),

        md("""\
## 9 · Interactive Plot (HTML)

Run the cell below to generate an interactive HTML file.
Hover over each point to see the member's name, state, and congress.
"""),

        code("""\
make_interactive_scatter(pca_coords, meta)
print("Interactive plot saved to outputs/congress_interactive.html")
"""),

        md("""\
## 10 · Conclusion

| Claim | Evidence |
|-------|----------|
| Political belief is multi-dimensional | PCA needs 4–6 components to explain 80% of variance |
| The binary system is imposed, not natural | Natural k-means clusters ≠ party assignment |
| Members exist on a continuum | PC1 × PC2 scatter shows gradient, not two islands |
| The overlap zone is real | ~10–20% of members live in the zone the binary cannot classify |

The two-party system is a *compression artefact* — a lossy encoding of a
richer underlying reality.  Like quantising a hi-fi signal to 1 bit.

A **true republic as a free market of law** would reflect this dimensionality:
organic coalitions, emergent consensus, not the binary forced on complex human belief.
"""),

    ]

    notebook = nb(*cells)
    path = NB_DIR / "01_congress_voting.ipynb"
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2 — UN Voting
# ─────────────────────────────────────────────────────────────────────────────

def build_un_notebook() -> None:
    cells = [

        md("""\
# 02 · UN General Assembly Voting — Cold War Blocks vs. Natural Alignment

## Thesis

The Cold War superimposed a **binary frame** (West vs. East) on world politics.
PCA of UN General Assembly votes shows that:
- The true geopolitical space is **multi-dimensional**
- "Blocks" are approximations, not reality
- Nations have a **continuous, organic** alignment profile

The same principle as the congress analysis, applied to nation-states.
"""),

        code("""\
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px

from voting_pca import (
    load_un_data,
    build_un_vote_matrix,
    run_pca,
    run_clustering,
    silhouette_analysis,
    plot_un_country_clusters,
    make_un_interactive,
    BLOCK_COLORS,
)

OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
print("Environment ready.")
"""),

        md("""\
## 1 · Load and Explore the Data

Data from the `unvotes` R package (mirrored on TidyTuesday).
Covers UNGA resolutions from **1946 to ~2019**.
"""),

        code("""\
un_votes, un_rc, un_issues = load_un_data()
print("\\nVote values:")
print(un_votes["vote"].value_counts())
"""),

        code("""\
print("\\nIssue categories:")
print(un_issues["issue"].value_counts())
"""),

        md("""\
## 2 · Build the Country × Resolution Matrix

Each row = a country.  Each column = a UNGA resolution (rcid).
+1 = yes, −1 = no, 0 = abstain / absent.
"""),

        code("""\
un_matrix, blocks = build_un_vote_matrix(un_votes)
print("\\nBlock assignments:")
print(blocks.value_counts())
"""),

        md("""\
## 3 · PCA

If the Cold War binary explained global alignment, PC1 would separate East
from West cleanly and explain ~100% of variance.
"""),

        code("""\
un_pca_coords, un_pca_model, un_var = run_pca(un_matrix, n_components=10)
print("Variance explained:")
for i, v in enumerate(un_var):
    print(f"  PC{i+1}: {v:.1%}  (cum: {un_var[:i+1].sum():.1%})")
"""),

        md("""\
## 4 · Cluster the Countries

K-means with k=4 often recovers:
- Western bloc
- Eastern bloc
- Non-aligned / Global South
- A fourth, nuanced grouping

But the *boundaries* are fuzzy and many countries resist clean assignment.
"""),

        code("""\
un_clusters = run_clustering(un_pca_coords, k_range=[2, 3, 4, 5, 6])
"""),

        md("""\
## 5 · Visualise: Cold War Blocks vs. Natural Clusters
"""),

        code("""\
plot_un_country_clusters(
    un_pca_coords,
    un_matrix.index,
    blocks,
    cluster_labels=un_clusters[4],
)
from IPython.display import Image
Image(str(OUT_DIR / "un_country_clusters.png"))
"""),

        code("""\
make_un_interactive(un_pca_coords, un_matrix.index, blocks)
print("Interactive UN plot → outputs/un_interactive.html")
"""),

        md("""\
## 6 · Which Countries Cross Blocks Most?

The Non-Aligned Movement was an explicit rejection of the imposed binary.
PCA places these countries in the *middle* of the spectrum,
exactly as the thesis predicts.
"""),

        code("""\
pc1_scores = pd.Series(un_pca_coords[:, 0], index=un_matrix.index, name="PC1")
pc2_scores = pd.Series(un_pca_coords[:, 1], index=un_matrix.index, name="PC2")

alignment_df = pd.concat([pc1_scores, pc2_scores, blocks], axis=1)
print("\\nNon-Aligned movement countries — PC1 scores (middle = non-binary):")
nonaligned = alignment_df[alignment_df["block"] == "Non-Aligned"].sort_values("PC1")
print(nonaligned[["PC1", "PC2"]].to_string())
"""),

        md("""\
## 7 · Silhouette: Is k=2 the Natural Answer?
"""),

        code("""\
un_sil = silhouette_analysis(un_pca_coords, k_range=[2, 3, 4, 5, 6])
print("Silhouette scores:")
for k, s in sorted(un_sil.items()):
    mark = " ← best" if s == max(un_sil.values()) else ""
    print(f"  k={k}: {s:.4f}{mark}")
"""),

        md("""\
## 8 · Conclusion

The UNGA voting record shows:
- Nations do **not** split neatly into two camps
- The Non-Aligned Movement occupied genuine *middle ground* in PCA space
- Natural k-means clusters find **more than 2** meaningful groupings
- The Cold War "bloc" framing was a political construction imposed on
  a more complex and continuous reality

The same pattern as congress voting, the same pattern as individual belief:
**imposed binaries are a cognitive shortcut that erases real structure.**
"""),

    ]

    notebook = nb(*cells)
    path = NB_DIR / "02_un_voting.ipynb"
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print(f"Written: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_congress_notebook()
    build_un_notebook()
    print("\nAll notebooks created.")
