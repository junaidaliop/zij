# MuCon

Implements MuCon, a clipped-Muon variant that bounds the spectral norm of the update direction.

Muon-style optimizers take a matrix-valued momentum (or preconditioned update) $B_t$ and replace it with its canonical partial polar factor $UV^\top$, which maps every nonzero singular value to one. MuCon instead applies singular-value clipping to the same Muon matrix: it leaves singular values at or below a threshold $\tau_t$ untouched and lowers only the violating directions to $\tau_t$. The clipped direction is the projection of $B_t$ onto the spectral-norm ball $\{X : \lVert X\rVert_2 \le \tau_t\}$, so $\lVert D_t^{\mathrm{MuCon}}\rVert_2 \le \tau_t$. Taking $\tau_t \to 0$ recovers gradient-descent-like behavior and, in the limit of saturated spectra, the clip reduces to Muon's polar step.

Writing the compact SVD $B_t = U\Sigma V^\top$ with $\Sigma = \mathrm{diag}(\sigma_1,\dots,\sigma_r)$, the parameter update replaces Muon's polar direction with the clipped one:

$$
\begin{aligned}
D_t^{\mathrm{Muon}} &= \mathrm{Polar}(B_t) = UV^\top \\
D_t^{\mathrm{MuCon}} &= \mathrm{MClip}_{\tau_t}(B_t) = U\,\mathrm{diag}(\min\{\sigma_i, \tau_t\})\,V^\top \\
\theta_t &= \theta_{t-1} - \eta\, D_t^{\mathrm{MuCon}}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $B_t$ the matrix-valued Muon momentum, $U,\Sigma,V$ its compact SVD with singular values $\sigma_i$, and $\tau_t > 0$ the per-step clipping threshold (default $\tau = 1$). Equivalently $\mathrm{MClip}_{\tau}(M) = \mathrm{argmin}_{\lVert X\rVert_2 \le \tau} \tfrac{1}{2}\lVert X - M\rVert_F^2$, the Frobenius projection onto the spectral-norm ball.

Reference: Albert Yi, "MuCon: Clipped Muon Updates for LLM Training", arXiv 2026. https://arxiv.org/abs/2605.26459

---
[Back to the Canon](../README.md)
