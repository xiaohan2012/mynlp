# TODO

1. PCA on higher dimensions(1000 instead of 50)?
2. Performance tuning? sparse row matrix or column matrix
3. Continue on HOMER

# label_distance_matrix.py

To see the effects of different distance metrics on the label clustering process, I computed the pairwise label distance and found the following metrics performing good:

- cosine
- correlation
- jaccard
- braycurtis

See the code for more details.

# exp_util.py

Utility that fit data and evaluate the result


# Experiments

## Distance metirc comparison

Refer to `result/distance_metric_and_dimension_reduction.txt`

I compared `cosine` and `euclidean`. Subjectivley, `cosine` produces better Clustering.

For the final score, `cosine` performs slightly better than `euclidean`

## Dimension reduction methods comparison

Refer to `result/distance_metric_and_dimension_reduction.txt`

`PCA` is much worse than `Random Projection`

## Comparison between BR and HOMER

Refer to `result/distance_metric_and_dimension_reduction.txt`

Possible reasons why HOMER is worse:

- The label frequency distribution is parse. There are quite a few labels with only one example. Refer to `figures/figures/sector_label_frequency_histogram.png`

Then maybe we can try *Classifier Chains for Multi-label Classification*, which fights the sparsity problem while considers some level of label dependency.

## Comparing BR and HOMER on delicious dataset

Applied `LinearSVC` and `GaussianNB` on `HOMER` and `Binary Relevance` method. See `result/*.txt` for results.

Conclusion:

- BR is better than HOMER.
- Performance of BR is not as bad as the on described in the original paper

If I try to replicate the experiment in the original paper, which uses `BernoulliNB`, the result is `HOMER` is better than `BR`.

So, the selection of classifier matters as well and `LinearSVC` is generally better.

Comment about the experiment in original paper:

It's not comprehensive and maybe more methods and more data sets should be used.

# Thinking:
1. What the possibly reasons that BR is better?
2. When is `HOMER` better and when is `BR` better?






