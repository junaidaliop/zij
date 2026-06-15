# AOFGD

Implements AOFGD, a fractional gradient descent method whose fractional order self-adapts during training.

Fractional gradient descent (FGD) replaces the integer-order gradient with a Caputo fractional-order derivative of order $\alpha$, where $\alpha$ controls how much past gradient history is folded into each step. Existing variable-order schemes vary $\alpha$ on a fixed schedule tied to the iteration count, which cannot react to the network's actual training state. AOFGD instead drives the order from a convergence evaluation factor computed online, so $\alpha$ adapts to the current optimization dynamics. This keeps the fast early convergence of FGD while improving final precision and removing the need to hand-tune the order; the authors report it integrating into Caputo-based fractional optimizers and outperforming both integer-order and fixed-order fractional methods across optimizers, datasets, and network architectures.

Reference: Kemeng Xiang, Chunna Zhao, Qian Su, Xiaojun Zhou, Junjie Ye, Yaqun Huang, "AOFGD: Adaptive order fractional gradient descent method", SSRN 2025. https://doi.org/10.2139/ssrn.5717167

---
[Back to the Canon](../README.md)
