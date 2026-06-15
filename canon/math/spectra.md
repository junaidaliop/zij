# SPECTRA

Implements SPECTRA, spectral clipping applied to a base optimizer's update matrix.

SPECTRA wraps an existing optimizer (AdamW, Signum, AdEMAMix) by clipping the spectral norm of its update before applying it. Given the base update matrix $U_t$, the compact SVD $U_t = P S Q^\top$ is formed and each singular value is capped at a threshold $c_t$, bounding the spectral norm of the step. The result is then scaled and applied with decoupled weight decay.

Because the exact SVD is costly, SPECTRA replaces hard clipping with a soft variant $H_{c}(U_t) = (I + U_t U_t^\top / c^2)^{-1/2} U_t$, computed through Newton-Schulz iterations using only matrix multiplications.

$$
\begin{aligned}
\mathrm{clip}_c(s) &= \mathrm{sign}(s)\,\min(|s|, c), \\
\mathrm{clip}^{\mathrm{sp}}_{c}(U_t) &= P\,\mathrm{diag}\big(\mathrm{clip}_c(S_{11}),\dots,\mathrm{clip}_c(S_{qq})\big)\,Q^\top, \quad U_t = P S Q^\top, \\
\theta_{t+1} &= (1 - \lambda \eta_t)\,\theta_t - \alpha \eta_t\, \mathrm{clip}^{\mathrm{sp}}_{c_t}(U_t).
\end{aligned}
$$

where $\theta$ are the matrix-shaped parameters, $\eta_t$ the learning rate, $\lambda$ the decoupled weight decay, $\alpha$ a scaling factor, $c_t$ the spectral clipping threshold, and $U_t$ the update matrix produced by the base optimizer.

Reference: Xiaowen Jiang, Andrei Semenov, Sebastian U. Stich, "Enhancing LLM Training via Spectral Clipping", arXiv 2026. https://arxiv.org/abs/2603.14315

---
[Back to the Canon](../README.md)
