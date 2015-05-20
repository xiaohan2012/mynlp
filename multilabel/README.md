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

See the code for more details


# Current server
ukko141


# Notes & Observations

1. Random projection is better than PCA
2. Cosine is slightly better than Euclidean

See `results.txt` for more.


# Thinking:
1. What the possibly reasons that BR is better?

For each classifier, BR uses more data than HOMER.

Maybe I should read more on how the algorithms splits the data
