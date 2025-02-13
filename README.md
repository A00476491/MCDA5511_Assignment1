## Instructions for use

1. Collect or format your data in the following format

| Name  | What are your interests? (or varying permutations of this question) |
| ----- | ------------------------------------------------------------------- |
| Alice | I love being the universal placeholder for every CS joke ever       |
| Bob   | I too love being the universal placeholder for every CS joke        |

2. Clone the repository
3. Install all required packages using pip or conda:

- `umap-learn`
- `scikit-learn`
- `scipy`
- `sentence-transformers`
- `matplotlib`
- `pyvis`
- `pandas`
- `numpy`
- `seaborn`
- `branca`

4. Run all cells
5. Below is a visualization where each point represents a person's interests, and the distance between points reflects the similarity of interests.
   ![visualization](visualization.png)


## Dimension Reduction Analysis

### Experiment Results

| Seed  | 42   | 52   | 62   |
|-------|------|------|------|
| Default Parameter (Rank Correlation) | 0.21 | 0.21 | 0.31 |
| Best Parameter (Rank Correlation)    | 0.70 | 0.51 | 0.61 |

With the default parameters, the **rank correlation** between embeddings before and after dimensionality reduction is **0.21** when `seed=42`.  
After parameter optimization, this increases significantly to **0.70**, effectively capturing the desired patterns, demonstrating the effectiveness of parameter tuning.

For `seed=52` and `seed=62`, the rank correlation with default parameters is **0.21** and **0.31**, respectively.  
Using the optimized parameters (originally tuned for `seed=42`), the correlation improves to **0.51** and **0.61**, showing that the optimization generalizes well across different seeds.

From the UMAP plots, the optimized UMAP clearly forms stable clusters, whereas the default settings fail to do so. This visually confirms the effectiveness of the optimization.

