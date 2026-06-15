# AdaMuon

Implements AdaMuon, an adaptive variance-normalized Muon optimizer.

AdaMuon augments Muon with an element-wise second moment applied to the
orthogonalized update. For a matrix parameter, the momentum $M_t$ is
orthogonalized through a Newton-Schulz iteration, a per-element second
moment $V_t$ is accumulated on the orthogonalized direction, and the
direction is variance-normalized before an RMS-aligned rescaling that
matches the update magnitude to Adam:


$$
\begin{aligned}
M_t &= \beta_1 M_{t-1} + (1 - \beta_1) G_t \\
O_t &= \mathrm{NewtonSchulz}(M_t) \\
V_t &= \beta_2 V_{t-1} + (1 - \beta_2)\, O_t \odot O_t \\
\hat{O}_t &= O_t \oslash \left(\sqrt{V_t / (1 - \beta_2^t)} + \epsilon\right) \\
\theta_t &= \theta_{t-1} - \gamma\,
    \frac{0.2 \sqrt{mn}}{\lVert \hat{O}_t \rVert_F}\, \hat{O}_t
\end{aligned}
$$

where $m, n$ are the matrix dimensions and $\odot$, $\oslash$
denote element-wise product and division. Parameters in a group with
`use_muon=False` are updated with decoupled-weight-decay AdamW instead, so
embeddings, heads, and scalar or vector parameters can share the optimizer.

This implementation follows kozistr/pytorch_optimizer and omits the paper's
$\mathrm{Sign}(M_t)$ sign-stabilization step before Newton-Schulz; that
is, it computes $O_t = \mathrm{NewtonSchulz}(M_t)$ rather than the
paper's $O_t = \mathrm{NewtonSchulz}(\mathrm{Sign}(M_t))$.

Unlike the paper, which applies no bias correction on $V_t$ (the
RMS-alignment rescale removes it), this implementation (following kozistr)
applies second-moment bias correction via $1 - \beta_2^t$. This factor
is cancelled by the subsequent RMS rescale, so the resulting update is
numerically unchanged.

Reference: Chongjie Si, Debing Zhang, Wei Shen, "AdaMuon: Adaptive Muon
Optimizer", 2025.
https://arxiv.org/abs/2507.11005

---
[Back to the Canon](../README.md)
