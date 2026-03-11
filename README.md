# PCA Living Systems — Imposed Binary vs. Natural Structure

## Conceptual Thesis

Political systems forced into **2-party blocks** mask the true underlying
**continuous, multi-dimensional** nature of political belief.

PCA and unsupervised clustering reveal the natural groupings that exist
*independently of imposed labels* — like a living system vs. a rigid binary.
This supports the idea of a **"true republic"** as a free market of law:
emergent and organic, not artificially constrained.

---

## Project Structure

```
PCA_Living_systems/
  data/              Raw CSVs (downloaded automatically)
  notebooks/         Jupyter notebooks with narrative analysis
  src/               Reusable Python modules
  outputs/           All generated plots and interactive HTML files
  requirements.txt
  README.md
```

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| HSall_members.csv | voteview.com | Congress members + DW-NOMINATE ideology scores |
| HSall_votes.csv | voteview.com | Individual roll-call votes (cast codes) |
| HSall_rollcalls.csv | voteview.com | Roll call metadata |
| un_votes.csv | TidyTuesday / dgrtwo/unvotes | UNGA country votes |
| un_roll_calls.csv | TidyTuesday | UNGA resolution metadata |
| un_roll_call_issues.csv | TidyTuesday | Issue-area tags |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data + generates all plots)
python src/run_analysis.py

# 3. (Optional) Build Jupyter notebooks
python src/build_notebooks.py
jupyter notebook notebooks/
```

## Outputs Generated

| File | Description |
|------|-------------|
| `congress_pca_by_party.png` | PC1×PC2 scatter colored by party label — shows continuous spectrum |
| `congress_natural_clusters.png` | Same plot colored by K-means clusters (k=3,4,5) |
| `congress_party_vs_natural_alignment.png` | Stacked bars: party composition of each natural cluster |
| `congress_pca_variance_explained.png` | Scree plot — how many dimensions does political space need? |
| `congress_crossover_members.png` | Members in the overlap zone the binary cannot classify |
| `congress_silhouette.png` | Silhouette scores — is k=2 really the natural answer? |
| `congress_interactive.html` | Interactive scatter (hover = member name/state/congress) |
| `un_country_clusters.png` | Cold War blocks vs. natural country groupings |
| `un_interactive.html` | Interactive UN scatter |

## Key Findings

1. **PCA needs 4–6 dimensions to explain 80% of congressional voting variance.**
   A truly binary system would need only 1.

2. **Natural K-means clusters (k=3–5) cut across party lines.**
   Clusters are not pure R or D — they mix members from both parties.

3. **~10–20% of members live in the PC1 overlap zone** between the party
   ideal points. The binary label cannot classify them meaningfully.

4. **UN General Assembly voting is similarly multi-dimensional.**
   Cold War "blocks" were a political construction imposed on a richer reality.

## Source Modules

- `src/download_data.py` — Downloads all CSVs with progress reporting
- `src/voting_pca.py` — All analysis functions (PCA, clustering, plotting)
- `src/run_analysis.py` — End-to-end pipeline runner
- `src/build_notebooks.py` — Generates .ipynb files programmatically
