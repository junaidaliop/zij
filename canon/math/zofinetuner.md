# ZO Fine-tuner

Implements ZO Fine-tuner, a learned zeroth-order optimizer that shapes the perturbation covariance with a small auxiliary network.

ZO Fine-tuner fine-tunes large language models with forward passes only, building on the SPSA estimate used by MeZO: perturb the parameters by $\pm\epsilon u_t$ along a random direction $u_t$, evaluate the loss at both points, and use the finite-difference quotient as a scalar coefficient on $u_t$. The novelty is that $u_t$ is no longer drawn from the isotropic $\mathcal{N}(0, I_d)$; instead its covariance $\Sigma_t$ is a learned, block-diagonal matrix produced by a lightweight per-block network from optimization features such as the recent loss and per-block gradient statistics.

To keep the perturbation magnitude decoupled from the learning rate, the covariance is renormalized at every step so that its Frobenius norm matches that of the identity, $\lVert \Sigma_t \rVert_F = \lVert I_d \rVert_F$ (equivalently $\lVert \Sigma_t \rVert_F^2 = d$). The estimated gradient is then applied with a plain descent step; no momentum or second-moment preconditioner is accumulated.

$$
\begin{aligned}
\Sigma_t &= \mathrm{diag}\!\left(\sigma_t^{(1)} I_{d_1},\ \ldots,\ \sigma_t^{(n)} I_{d_n}\right), \qquad \sigma_t^{(i)} = \mathrm{PertNN}^{(i)}(\,\cdot\,;\,\omega^{(i)}) \\
u_t &\sim \mathcal{N}\!\left(0,\ \frac{\lVert I_d \rVert_F}{\lVert \Sigma_t \rVert_F}\,\Sigma_t\right) \\
\hat{g}_t &= \frac{\mathcal{L}(\theta_t + \epsilon u_t;\, B_t) - \mathcal{L}(\theta_t - \epsilon u_t;\, B_t)}{2\epsilon}\, u_t \\
\theta_{t+1} &= \theta_t - \eta\, \hat{g}_t
\end{aligned}
$$

where $\theta \in \mathbb{R}^d$ are the parameters partitioned into $n$ blocks of sizes $d_1, \ldots, d_n$, $\mathcal{L}(\theta; B)$ is the loss on minibatch $B$, $u_t$ is the perturbation drawn from the learned and norm-matched covariance $\Sigma_t$, $\sigma_t^{(i)}$ is the per-block variance emitted by the auxiliary network $\mathrm{PertNN}^{(i)}$ with parameters $\omega^{(i)}$, $\epsilon$ is the perturbation scale, $\hat{g}_t$ is the resulting zeroth-order gradient estimate, and $\eta$ is the learning rate.

Reference: Kairun Zhang, Haoyu Li, Yanjun Zhao, Yifan Sun, Huan Zhang, "Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs", arXiv preprint 2025. https://arxiv.org/abs/2510.00419

---
[Back to the Canon](../README.md)
