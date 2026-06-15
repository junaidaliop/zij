# WSBD

Implements WSBD (Weighted Stochastic Block Descent), a freezing-based wrapper that updates only an importance-weighted random subset of parameters each step.

Training quantum neural networks is hampered by barren plateaus and noisy gradients, where blindly updating every parameter wastes evaluations. WSBD accumulates each parameter's gradient over a window of $\tau$ steps to form an importance score, normalizes those scores into a selection distribution, and samples a fraction $1-\lambda_f$ of the parameters (without replacement) to remain active while the rest are frozen for that step. The chosen parameters are then advanced by the base optimizer's update $u_t$ (SGD, Adam, etc.), gated through a binary mask.

$$
\begin{aligned}
\mathcal{I}_p(\theta_k) &= \left| \sum_{t=1}^{\tau} \frac{\partial \mathcal{C}(\theta^{(t)}, x_t)}{\partial \theta_k} \right| + \epsilon \\
p_k &= \frac{\mathcal{I}_p(\theta_k)}{\sum_{i=1}^{|\theta|} \mathcal{I}_p(\theta_i)} \\
N_{\mathrm{active}} &= \left\lceil (1 - \lambda_f)\,|\theta| \right\rceil, \quad \delta^{(t)} = \mathrm{mask}\big(\mathrm{sample}(p, N_{\mathrm{active}})\big) \\
\theta^{(t+1)} &= \theta^{(t)} - \eta_t \left( \delta^{(t)} \odot u_t \right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ is the learning rate, $\mathcal{C}$ is the cost, $\mathcal{I}_p(\theta_k)$ is the accumulated importance of parameter $\theta_k$ over the window $\tau$, $p_k$ is its selection probability, $\lambda_f \in [0,1)$ is the freezing fraction, $N_{\mathrm{active}}$ is the number of parameters kept active, $\delta^{(t)} \in \{0,1\}^{|\theta|}$ is the binary mask formed by sampling $N_{\mathrm{active}}$ parameters without replacement according to $p$, $u_t$ is the base optimizer's update vector, $\odot$ is element-wise multiplication, and $\epsilon$ is a small constant for stability.

Reference: Christopher Kverne, Mayur Akewar, Yuqian Huo, Tirthak Patel, Janki Bhimani, "WSBD: Freezing-Based Optimizer for Quantum Neural Networks", arXiv 2026. https://arxiv.org/abs/2602.11383

---
[Back to the Canon](../README.md)
