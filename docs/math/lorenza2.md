# LORENZA

Implements LORENZA, a memory-efficient zeroth-order adaptive sharpness-aware optimizer for low-rank LLM training.

LORENZA combines low-rank gradient compression with sharpness-aware minimization (SAM), but estimates the SAM ascent perturbation using only forward passes. Every $T$ steps it picks a rank-$r$ subspace from a randomized SVD of the full gradient, giving projection matrices $Q_t$ and $R_t$. Within that subspace it forms a perturbation direction via a zeroth-order finite-difference estimate over random low-rank probes, takes a SAM step at the perturbed weights, projects the SAM gradient back into the subspace, and runs an Adam update there before lifting the step to full dimension.

$$
\begin{aligned}
G_t^{\mathrm{Pert}} &= -\frac{1}{q}\sum_{j=1}^{q}\frac{f(\theta_t + \mu P_j) - f(\theta_t - \mu P_j)}{2\mu}\,P_j, \quad P_j = Q_t\,\mathrm{diag}(u_j)\,R_t \\
G_t^{\mathrm{SAM}} &= \nabla_\theta f\!\left(\theta_t + \rho\,\frac{G_t^{\mathrm{Pert}}}{\lVert G_t^{\mathrm{Pert}}\rVert_F}\right) \\
\hat{g}_t &= Q_t^{\top} G_t^{\mathrm{SAM}} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\hat{g}_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat{g}_t^{2} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_{t+1} &= \theta_t - \gamma\,Q_t\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\end{aligned}
$$

where $Q_t \in \mathbb{R}^{m\times r}$, $R_t \in \mathbb{R}^{r\times n}$ are the low-rank projection matrices from randomized SVD of the gradient, $u_j \sim \mathcal{N}(0, I_r)$ are random probe vectors, $\mu$ is the finite-difference smoothing radius, $q$ the number of probes, $\rho$ the SAM neighborhood size, $\gamma$ the learning rate, $\beta_1,\beta_2$ the Adam decay rates, and $\epsilon$ the stability constant. Moments are tracked in the $r$-dimensional subspace, reducing optimizer memory from $O(mn)$ to $O(mr)$.

Reference: Yehonathan Refael, Iftach Arbel, Ofir Lindenbaum, Tom Tirer, "LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training and Fine-Tuning via Efficient Zeroth-Order Adaptive SAM Optimization", ICML 2025. https://arxiv.org/abs/2502.19571

---
[Back to the Canon](../index.md)
