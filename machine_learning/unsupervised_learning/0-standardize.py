#!/usr/bin/env python3
"""Standardization module."""

from sklearn.preprocessing import StandardScaler


def Standardize(X):
    """Standardize data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
