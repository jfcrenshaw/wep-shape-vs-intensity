"""
1. Simulate donuts with random perturbations in sparse Zernikes
2. Keep only shape information
3. Estimate wavefront using Danish
"""

import numpy as np
from lsst.ts.wep.utils import forwardModelPair
from lsst.ts.wep.estimation import WfEstimator

import pickle

j = np.arange(4, 23)
nu = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 2])
norms = 10.0 ** (-6 - nu / 2) / (j - 3) ** (0.5)

# Create wavefront estimator
nollIndices = np.array([4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20, 21])
wfEst = WfEstimator(algoName="danish", nollIndices=nollIndices)

# Create dictionary to hold all Zernikes
zk = {"true": [], "est": []}

# 100 different simulations
seed = 0
while len(zk["est"]) < 100:
    # Create RNG
    seed += 1
    rng = np.random.default_rng(seed)

    # Generate random Zernikes
    zkCoeff = 0.0 * np.arange(4, 23)
    _zkCoeff = rng.normal(0, norms)
    _zkCoeff = np.clip(_zkCoeff, -1e-6, +1e-6)
    zkCoeff[nollIndices - 4] = _zkCoeff[nollIndices - 4]

    # Random field angle
    fieldAngleRadius = 0.5
    fieldAngleAzimuth = rng.uniform(0, 2 * np.pi)
    fieldAngle = fieldAngleRadius * np.array(
        [np.cos(fieldAngleAzimuth), np.sin(fieldAngleAzimuth)]
    )

    try:
        # Forward model images
        _, intra, extra = forwardModelPair(
            seed=seed,
            zkCoeff=zkCoeff,
            fieldAngleIntra=fieldAngle,
            seeing=0.5,
            skyLevel=100,
            flat=True,
        )

        # Skip if one of the images isn't finite
        if not np.isfinite(intra.image).all() or not np.isfinite(extra.image).all():
            continue

        # Estimate Zernikes
        _zkEst = wfEst.estimateZk(intra, extra)
        zkEst = np.zeros_like(zkCoeff)
        zkEst[nollIndices - 4] = _zkEst

        # Append Zernikes to lists (in microns)
        zk["true"].append(zkCoeff * 1e6)
        zk["est"].append(zkEst)
    except:
        continue

    # Estimate Zernikes
    _zkEst = wfEst.estimateZk(intra, extra)
    zkEst = np.zeros_like(zkCoeff)
    zkEst[nollIndices - 4] = _zkEst

    # Append Zernikes to lists (in microns)
    zk["true"].append(zkCoeff * 1e6)
    zk["est"].append(zkEst)

# Convert lists to arrays
zk["true"] = np.array(zk["true"])
zk["est"] = np.array(zk["est"])

# Save in pickle file
with open("sims/sparse_zk_estimates_shape.pkl", "wb") as file:
    pickle.dump(zk, file)
