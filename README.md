# Introduction
This repository explores embeddings—what they are, how they change across different datasets and models, and how they appear in a 2D space. We use an interest dataset to gain insights into embeddings. The dataset contains sentence-level interests, all collected from real people.

## Instructions for Use

1. **Clone the repository**
2. **Create the environment**
   ```bash
   conda env create -f environment.yml
   conda activate 5511assignment1
   ```
3. **Run code**  
   Below is a visualization where each point represents a person's interests, and the distance between points reflects the similarity of interests.
   
   ![visualization](visualization.png)

---

## Task1: What Are Embeddings
### Concept of Embedding
Embedding is a kind of numerical representation of text (words, phrases, or sentences), enabling computers to understand and process natural language.
Its core idea is that texts with similar meanings should be closer together in the numerical space.

Think of each word or sentence as a unique point on a map, where words with similar meanings are located near each other.
For easy example, the words “happy” and “joyful” will be close together, while “sad” will be farther away. Similarly, 
whole sentences with related meanings will be positioned near each other in this space.

In the visualization graph, we can see Anuja and Max are close to each other because they both mentioned AI models.
### How Embeddings Capture the Meaning of Words?
Embeddings convert words into numbers so that computers can process them. 
But these aren’t just random numbers—they are carefully designed so that similar words get similar number patterns.

### key Machine Learning Techniques for embeddings:
Word2Vec:  **Word2Vec learns word relationships by predicting words in a sentence.**

 Two methods:

`Skip-Gram:` Predicts surrounding words given a target word.

`CBOW (Continuous Bag of Words)`: Predicts the target word from surrounding words.



## Task3: Embedding sensitivity analysis (differrent models)

### Experiment Results

   Spearman’s rank correlation coefficient is mainly used to measure the monotonicity between two ranking sequences.

   The value range is [-1, 1]:
      
      • 1 means that the two rankings are consistent in monotonicity.

      • 0 means that the two rankings are independent in monotonicity.

      • -1 means that the two rankings are completely opposite in monotonicity.

   The results are all close to “correlation=0.86”. From my(SicongFu's) perspective, This shows that the correlation between these rankings (all-MiniLM-L6-v2 with all-MiniLM-L12-v2 or all-mpnet-base-v2) is high, which means that in judging the text similarity, the output results of these models are relatively close.

### Analyze the causes 

   Different models differ in structural depth, number of parameters, and pre-training corpus. Taking MiniLM and MPNet as examples, they use different network structures, number of layers, and language features learned on large-scale corpora, which leads to different ways of capturing text semantics. It is this difference in the underlying mechanism that often gives different results when inferring similarity or ranking.

   The amount of text used in the current test may be relatively limited, and personal description texts are often short and diverse in content. Due to the small amount of data and different text styles, any subtle differences in feature extraction may be magnified, resulting in inconsistent weights assigned to specific sentences or keywords by different models, which greatly affects the order of similarity ranking.

   Different models do not judge "similarity" in exactly the same way. Some models may pay more attention to keyword entities, while others tend to capture semantic integrity or context associations. Therefore, even for the same text, different models may have different "focus points". When the amount of data is relatively small and the text content varies greatly, this difference in "focus points" often makes it difficult to maintain consistent ranking results.
   
## Task4: Dimension Reduction Analysis

### Experiment Results

| Seed  | 42   | 52   | 72   |
|-------|------|------|------|
| Default Parameter | 0.21 | 0.44 | 0.11 |
| Best Parameter    | 0.64 | 0.50 | 0.53 |

(Note: The values represent Rank Correlation.)

### Findings
With the default parameters, the **rank correlation** between embeddings before and after dimensionality reduction is **0.21** when `seed=42`.  
After parameter optimization, this increases significantly to **0.64**, effectively capturing the desired patterns, demonstrating the effectiveness of parameter tuning.

For `seed=52` and `seed=62`, the rank correlation with default parameters is **0.44** and **0.11**, respectively.  
Using the optimized parameters (originally tuned for `seed=42`), the correlation improves to **0.50** and **0.53**, showing that the optimization generalizes well across different seeds.
