# Newton-Muon

Implements Newton-Muon, Muon with right-preconditioning by the inverse second moment of layer activations.

Muon orthogonalizes the momentum of a weight matrix's gradient via the matrix sign function $\mathrm{msgn}(X)=UV^\top$ (from the compact SVD $X=US V^\top$), approximated in practice by Newton-Schulz iterations. Newton-Muon augments this with a curvature correction: it right-multiplies the gradient by the inverse second moment of the layer inputs $Z$, $(ZZ^\top)^{-1}$, which acts as a Newton-style preconditioner along the input directions before the gradient enters the Muon pipeline.

In the idealized form the update is $W \leftarrow W - \eta\,\mathrm{msgn}\!\big(G (ZZ^\top)^{-1}\big)$. The practical algorithm keeps a damped EWMA estimate $K$ of the activation second moment, applies its ridge-regularized inverse to the raw gradient, and then runs the standard Muon momentum and orthogonalization.

$$
\begin{aligned}
K_t &= \beta_2\, K_{t-1} + (1-\beta_2)\, \tfrac{1}{N} Z_t Z_t^\top \\
\tilde{G}_t &= G_t \,\big(K_t + \gamma I\big)^{-1} \\
m_t &= \beta_1\, m_{t-1} + \tilde{G}_t \\
\theta_t &= \theta_{t-1} - \eta \,\mathrm{msgn}(m_t)
\end{aligned}
$$

where $\theta$ is the weight matrix $W$, $\eta$ the learning rate, $G_t$ the raw gradient, $Z_t$ the stacked layer-input activations over a batch of size $N$, $K_t$ the EWMA estimate of the activation second moment with decay $\beta_2$, $\gamma$ the ridge damping, $m_t$ the momentum buffer with decay $\beta_1$, and $\mathrm{msgn}(\cdot)$ the matrix sign function realized through Newton-Schulz iterations. The inverse $(K_t+\gamma I)^{-1}$ is refreshed every $k$ steps and applied to the raw gradient before momentum and weight decay.

Reference: Zhehang Du, Weijie Su, "The Newton-Muon Optimizer", 2025. https://arxiv.org/abs/2604.01472

---
[Back to the Canon](../README.md)
