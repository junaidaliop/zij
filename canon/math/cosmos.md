# COSMOS

Implements COSMOS, a hybrid adaptive optimizer that splits the gradient matrix into a leading eigensubspace handled SOAP-style and a complementary subspace handled Muon-style.

COSMOS exploits the observation that eigensubspaces of the gradient matrix carry uneven importance. For each matrix-shaped parameter it tracks a low-rank eigenbasis $U_t$ (rank $r \ll n$) of the gradient second moment, updated by a power-iteration-with-QR step. Inside this rank-$r$ subspace it applies a SOAP-like preconditioned update $A_t$ using a projected second moment $V_t$ with Adam-style bias correction. The residual energy outside the subspace, $M_t - M_t U_t U_t^\top$, is handled by a Muon-like update $B_t$: Frobenius normalization followed by Newton-Schulz orthogonalization. The two pieces are linearly combined and the whole step is rescaled to unit per-element scale before being applied, keeping memory at $O(mr + nr)$ per layer instead of the full $O(mn)$ preconditioner.

$$
\begin{aligned}
M_t &= \beta_1 M_{t-1} + (1-\beta_1)\, G_t \\
U_t &= \mathrm{QR}\big(\beta_2 U_{t-1} S_{t-1} + (1-\beta_2)\, G_t^\top G_t U_{t-1}\big) \\
S_t &= U_t^\top\big(\beta_2 U_{t-1} S_{t-1} U_{t-1}^\top + (1-\beta_2)\, G_t^\top G_t\big) U_t \\
V_t &= \beta_2 V_{t-1} + (1-\beta_2)\, (G_t U_t)\odot(G_t U_t) \\
A_t &= \left(\frac{M_t U_t / (1-\beta_1^{t})}{\sqrt{(V_t+\epsilon)/(1-\beta_2^{t})}}\right) U_t^\top \\
B_t &= \mathrm{NORM}\!\left(\mathrm{NS5}\!\left(\frac{M_t - M_t U_t U_t^\top}{\lVert M_t - M_t U_t U_t^\top\rVert_F}\right)\right) \\
\tilde{G}_t &= A_t + \gamma\, B_t \sqrt{m} \\
W_{t+1} &= W_t - \eta\, \mathrm{NORM}(\tilde{G}_t)
\end{aligned}
$$

where $W_t$ are the parameters, $G_t$ the gradient of shape $m \times n$, $M_t$ the first-moment EMA, $U_t \in \mathbb{R}^{n\times r}$ the leading eigenbasis with $S_t \in \mathbb{R}^{r\times r}$ its second-moment projection, $V_t$ the second moment within the subspace, $\eta$ the learning rate, $\gamma$ the combination weight, $\beta_1,\beta_2$ the moment decay rates, $\epsilon$ a stability constant, $\mathrm{NS5}(\cdot)$ five Newton-Schulz orthogonalization iterations, and $\mathrm{NORM}(X) = \sqrt{n}\, X / \lVert X\rVert_F$ the Frobenius rescaling.

Reference: Liming Liu, Zhenghao Xu, Zixuan Zhang, Hao Kang, Zichong Li, Chen Liang, Weizhu Chen, Tuo Zhao, "COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs", arXiv 2025. https://arxiv.org/abs/2502.17410

---
[Back to the Canon](../README.md)
