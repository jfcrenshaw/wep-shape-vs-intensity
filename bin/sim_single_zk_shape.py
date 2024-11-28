"""
1. Simulate donuts while perturbing single Zernikes
2. Keep only shape information
3. Estimate wavefront using Danish
"""
import numpy as np
from lsst.ts.wep.utils import forwardModelPair
from lsst.ts.wep.estimation import WfEstimator

import pickle


# Create wavefront estimator
wfEst = WfEstimator(algoName="danish")

# Create dictionary to hold all Zernikes
zk = {}

# Loop over Noll indices
for j in np.arange(4, 23):
    zk[j] = {"true": [], "est": []}
    
    # 100 different times
    seed = 0
    while len(zk[j]["est"]) < 100:
        # Create RNG
        seed += 1
        rng = np.random.default_rng(seed)
    
        # Generate random Zernikes
        zkCoeff = np.zeros_like(np.arange(4, 23), dtype=float)
        _zkCoeff = rng.normal(0, 1e-6 / np.arange(1, 20) ** 1.5)
        _zkCoeff = np.clip(_zkCoeff, -1e-6, +1e-6)
        zkCoeff[j - 4] = _zkCoeff[j - 4]

        # Random field angle
        fieldAngleRadius = 0.5
        fieldAngleAzimuth = rng.uniform(0, 2 * np.pi)
        fieldAngle = fieldAngleRadius * np.array(
            [np.cos(fieldAngleAzimuth), np.sin(fieldAngleAzimuth)]
        )

        # Forward model images
        try:
            _, intra, extra = forwardModelPair(
                seed=seed,
                zkCoeff=zkCoeff,
                fieldAngleIntra=fieldAngle,
                flat=True,
            )
        except:
            continue

        # Estimate Zernikes
        zkEst = wfEst.estimateZk(intra, extra)

        # Append Zernikes to lists (in microns)
        zk[j]["true"].append(zkCoeff * 1e6)
        zk[j]["est"].append(zkEst)

    # Convert lists to arrays
    zk[j]["true"] = np.array(zk[j]["true"])
    zk[j]["est"] = np.array(zk[j]["est"])

# Save in pickle file
with open("sims/single_zk_estimates_shape.pkl", "wb") as file:
    pickle.dump(zk, file)




        
        