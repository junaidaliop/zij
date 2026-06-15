# FedGMT

Implements FedGMT, a federated sharpness-aware method that measures global sharpness from the global model trajectory instead of an extra SAM backward pass.

FedGMT replaces SAM's perturbed-loss term with a *global model trajectory* loss. The server keeps an exponential moving average $e_t$ of the global model; a first-order argument shows that minimizing $L(e_t) - L(w_t)$ approximates SAM's sharpness measure $S(w_t) = L(w_t+\epsilon) - L(w_t)$ for the global objective. Each client therefore trains on its empirical risk plus a KL term that pulls its model toward the EMA model $e_t$, which costs one extra forward pass and no extra backward pass.

To keep local updates aligned with the global objective under data heterogeneity, FedGMT casts the constraint $w_m = w_t$ in an ADMM form: clients run local SGD on the trajectory loss with a dual correction $u_m$, then the server updates the dual variable and aggregates. For client $m$ at round $t$ with local step $k$:

$$
\begin{aligned}
e_t &= \alpha\, e_{t-1} + (1-\alpha)\, w_t \\
L_m(w_m, e_t) &= L_m(w_m) + \frac{\gamma}{|\mathcal{D}_m|} \sum_{\mathcal{D}_m} \ell_{\mathrm{KL}}\!\big(f(e_t),\, f(w_m)\big) \\
g_{m,k} &= \nabla L_m(w_{m,k},\, e_t) \\
w_{m,k+1} &= w_{m,k} - \eta\,(g_{m,k} - u_m) \\
u_m^{t+1} &= u_m^{t} - \tfrac{1}{\beta}\,(w_{m,K} - w_t) \\
w_{t+1} &= \frac{1}{|\mathcal{N}_t|} \sum_{m \in \mathcal{N}_t} \big(w_{m,K} - \beta\, u_m^{t+1}\big)
\end{aligned}
$$

where $w$ are the model parameters, $e_t$ the EMA (trajectory) model, $\eta$ the learning rate, $g_{m,k}$ the local gradient of the trajectory loss, $u_m$ the ADMM dual variable, $\alpha \in (0,1)$ the EMA coefficient, $\gamma$ the sharpness strength, $\beta$ the ADMM penalty coefficient, $\ell_{\mathrm{KL}}$ the KL divergence between the EMA and local model outputs $f(\cdot)$, $\mathcal{D}_m$ the client dataset, and $\mathcal{N}_t$ the active clients in round $t$.

Reference: Yuhang Li, Tong Liu, Yangguang Cui, Ming Hu, Xiaoqiang Li, "One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory", ICML 2025. https://openreview.net/forum?id=80mK2Mqaph

---
[Back to the Canon](../README.md)
