## Dimension Reduction Analysis

### Experiment Results

| Seed  | 42   | 52   | 72   |
|-------|------|------|------|
| Default Parameter | 0.21 | 0.44 | 0.11 |
| Best Parameter    | 0.64 | 0.50 | 0.53 |

(Note: The values represent Rank Correlation.)


With the default parameters, the **rank correlation** between embeddings before and after dimensionality reduction is **0.21** when `seed=42`.  
After parameter optimization, this increases significantly to **0.64**, effectively capturing the desired patterns, demonstrating the effectiveness of parameter tuning.

For `seed=52` and `seed=62`, the rank correlation with default parameters is **0.44** and **0.11**, respectively.  
Using the optimized parameters (originally tuned for `seed=42`), the correlation improves to **0.50** and **0.53**, showing that the optimization generalizes well across different seeds.

From the UMAP plots, the optimized UMAP cluster embeddings better, whereas the default settings fail to do so. This visually confirms the effectiveness of the optimization.
![UMAP Visualization](./vis/umap_comparison.png)

