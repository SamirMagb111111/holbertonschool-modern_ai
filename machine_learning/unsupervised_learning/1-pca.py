#!/usr/bin/env python3
"""PCA module."""

from sklearn.decomposition import PCA


def Apply_PCA(X, n_components, random_state):
    """Apply PCA to data."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    return X_pca, pca
