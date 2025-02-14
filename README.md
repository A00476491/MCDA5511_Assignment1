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

# For branch embedding-sensitivity-tests

## 1. Create the environment

   On Bash:

   conda env create -f environment1.yml

   conda activate branch__environment 

## 2. Analyze the results

   Spearman’s rank correlation coefficient is mainly used to measure the monotonicity between two ranking sequences.

   The value range is [-1, 1]:
      
      • 1 means that the two rankings are consistent in monotonicity.

      • 0 means that the two rankings are independent in monotonicity.

      • -1 means that the two rankings are completely opposite in monotonicity.

   The results are all close to “correlation=0.86”. From my(SicongFu's) perspective, This shows that the correlation between these rankings (all-MiniLM-L6-v2 with all-MiniLM-L12-v2 or all-mpnet-base-v2) is high, which means that in judging the text similarity, the output results of these models are relatively close.

## 3. Analyze the causes 

   Different models differ in structural depth, number of parameters, and pre-training corpus. Taking MiniLM and MPNet as examples, they use different network structures, number of layers, and language features learned on large-scale corpora, which leads to different ways of capturing text semantics. It is this difference in the underlying mechanism that often gives different results when inferring similarity or ranking.

   The amount of text used in the current test may be relatively limited, and personal description texts are often short and diverse in content. Due to the small amount of data and different text styles, any subtle differences in feature extraction may be magnified, resulting in inconsistent weights assigned to specific sentences or keywords by different models, which greatly affects the order of similarity ranking.

   Different models do not judge "similarity" in exactly the same way. Some models may pay more attention to keyword entities, while others tend to capture semantic integrity or context associations. Therefore, even for the same text, different models may have different "focus points". When the amount of data is relatively small and the text content varies greatly, this difference in "focus points" often makes it difficult to maintain consistent ranking results.
