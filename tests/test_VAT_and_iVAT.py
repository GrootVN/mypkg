import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from mypkg import VAT, iVAT, fastVAT

# --- 1. Create synthetic data (3 clusters) ---

X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.60,
    random_state=42
)

# Pairwise Euclidean distance matrix
D = pairwise_distances(X, metric="euclidean")

print("Distance matrix shape:", D.shape)

# --- 2. Run fastVAT on the distance matrix ---

RV_fast, p_fast = fastVAT(D, inplace=False)
print("fastVAT permutation:", p_fast[:10], "...")

# --- 3. Run classical VAT (slower but pure Python) ---

RV_vat, C_vat, I_vat, RI_vat = VAT(D)
print("VAT permutation I_vat:", I_vat[:10], "...")

# --- 4. Run iVAT (improved VAT) starting from original D ---

RiV, RV_from_iVAT = iVAT(D, VATflag=False, fastVATflag=True, cutflag=False)

# --- 5. Visualize original, VAT, and iVAT matrices ---

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original distance matrix
axes[0].imshow(D, cmap="gray", aspect="auto")
axes[0].set_title("Original distance matrix")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Index")

# VAT-ordered matrix (from fastVAT)
axes[1].imshow(RV_fast, cmap="gray", aspect="auto")
axes[1].set_title("VAT-ordered (fastVAT)")
axes[1].set_xlabel("VAT index")
axes[1].set_ylabel("VAT index")

# iVAT matrix
axes[2].imshow(RiV, cmap="gray", aspect="auto")
axes[2].set_title("iVAT matrix")
axes[2].set_xlabel("VAT index")
axes[2].set_ylabel("VAT index")

plt.tight_layout()
plt.show()
