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

   The results are all close to -0.04. From my(SicongFu's) perspective, the similarity rankings I get with all-MiniLM-L6-v2 are basically incompatible with the rankings of another model (all-MiniLM-L12-v2 or all-mpnet-base-v2).

## 3. Analyze the causes 

   Different models differ in structural depth, number of parameters, and pre-training corpus. Taking MiniLM and MPNet as examples, they use different network structures, number of layers, and language features learned on large-scale corpora, which leads to different ways of capturing text semantics. It is this difference in the underlying mechanism that often gives different results when inferring similarity or ranking.

   The amount of text used in the current test may be relatively limited, and personal description texts are often short and diverse in content. Due to the small amount of data and different text styles, any subtle differences in feature extraction may be magnified, resulting in inconsistent weights assigned to specific sentences or keywords by different models, which greatly affects the order of similarity ranking.

   Different models do not judge "similarity" in exactly the same way. Some models may pay more attention to keyword entities, while others tend to capture semantic integrity or context associations. Therefore, even for the same text, different models may have different "focus points". When the amount of data is relatively small and the text content varies greatly, this difference in "focus points" often makes it difficult to maintain consistent ranking results.