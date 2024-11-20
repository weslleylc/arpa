# Feature Selection Transformer

## Overview

The `FeatureSelectionTransformer` class is a transformer for feature selection using various information-theoretic measures.

## Parameters

- `k` (int): Number of features to select.
- `costs` (array-like, optional): Precomputed cost matrix.
- `function` (callable): Function to compute the cost between features.
- `precision` (int): Precision for discretization.
- `verbose` (int): Verbosity level.
- `processes` (int): Number of processes to use for computation.

## Methods

### `fit(X, y)`

Fit the transformer to the data.

### `transform(X)`

Transform the data to select the top `k` features.