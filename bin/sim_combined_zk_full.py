"""
1. Simulate donuts with random perturbations in all Zernikes
2. Keep surface brightness fluctuations
3. Estimate wavefront using Danish
"""
import numpy as np
from lsst.ts.wep.utils import forwardModelPair
from lsst.ts.wep.estimation import WfEstimator

import pickle


# Create wavefront estimator
wfEst = WfEstimator(algoName="danish")

# Create dictionary to hold all Zernikes
zk = {"true": [], "est": []}
    
# 100 different simulations
seed = 0
while len(zk["est"]) < 100:
    # Create RNG
    seed += 1
    rng = np.random.default_rng(seed)

    # Generate random Zernikes
    zkCoeff = rng.normal(0, 1e-6 / np.arange(1, 20) ** 1.5)
    zkCoeff = np.clip(zkCoeff, -1e-6, +1e-6)

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
            flat=False,
        )
    except:
        continue

    # Estimate Zernikes
    zkEst = wfEst.estimateZk(intra, extra)

    # Append Zernikes to lists (in microns)
    zk["true"].append(zkCoeff * 1e6)
    zk["est"].append(zkEst)

# Convert lists to arrays
zk["true"] = np.array(zk["true"])
zk["est"] = np.array(zk["est"])

# Save in pickle file
with open("sims/combined_zk_estimates_full.pkl", "wb") as file:
    pickle.dump(zk, file)




        
        