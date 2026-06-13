#!/usr/bin/env python3
"""PCA dimensionality reduction module."""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Apply PCA to tabular data using Scikit-learn."""
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )

    X_pca = pca.fit_transform(X)

    return X_pca, pca
