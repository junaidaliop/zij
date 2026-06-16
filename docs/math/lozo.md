# LOZO

Implements LOZO (Low-rank ZO), a zeroth-order fine-tuning method that perturbs each weight matrix along a low-rank random subspace.

Standard ZO estimators perturb the full parameter tensor with isotropic Gaussian noise, which is wasteful when the true gradient of a large language model has low-rank structure. LOZO instead perturbs each weight matrix $\theta$ with a rank-$r$ direction $UV^\top$ formed from two Gaussian factors, and estimates the gradient by symmetric finite differences along that direction. A lazy sampling scheme fixes $V$ for $\nu$ consecutive steps while resampling $U$ each step, so the iterates accumulate progress within a shared subspace before it is refreshed. The momentum variant LOZO-M stores the running average in the same low-rank factored form to keep the memory cost of a ZO method.

$$
\begin{aligned}
U_t &\sim \mathcal{N}(0, I)_{m\times r}, \qquad V_k \sim \mathcal{N}(0, I)_{n\times r}\ \ (\text{resampled every } \nu \text{ steps}), \\
c_t &= \frac{F(\theta_t + \epsilon\, U_t V_k^\top) - F(\theta_t - \epsilon\, U_t V_k^\top)}{2\epsilon}, \\
\hat{g}_t &= c_t\,\frac{U_t V_k^\top}{r}, \\
\theta_{t+1} &= \theta_t - \eta\, \hat{g}_t, \\
\text{(LOZO-M)}\quad N_t &= \beta\, N_{t-1} + (1-\beta)\, c_t\, U_t, \\
\theta_{t+1} &= \theta_t - \eta\, N_t \frac{V_k^\top}{r},
\end{aligned}
$$

where $\theta$ is a weight matrix of size $m\times n$, $\eta$ is the learning rate, $\epsilon$ is the perturbation (smoothing) radius, $r \ll \min\{m,n\}$ is the perturbation rank, $U_t V_k^\top$ is the low-rank perturbation direction, $c_t$ is the scalar finite-difference coefficient, $\hat{g}_t$ is the low-rank gradient estimate, $\nu$ is the lazy sampling interval (with $k = \lfloor t/\nu \rfloor$ indexing the current subspace), $N_t$ is the low-rank momentum factor, and $\beta$ is the momentum coefficient; on subspace transitions the old momentum is reprojected as $\tilde{N}_{t-1} = N_{t-1}\, V_k^\top V_{k+1}/n$.

Reference: Yiming Chen, Yuan Zhang, Liyuan Cao, Kun Yuan, Zaiwen Wen, "Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures", ICLR 2025. https://arxiv.org/abs/2410.07698

---
[Back to the Canon](../index.md)
