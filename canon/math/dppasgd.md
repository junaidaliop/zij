# DP-PASGD

Implements DP-PASGD, differentially private periodic-averaging SGD for federated learning across resource-constrained devices.

Each of the $M$ devices runs local noisy SGD on its own data, perturbing every gradient step with Gaussian noise to guarantee differential privacy. The local models are synchronized by averaging only once every $\tau$ iterations, which trades communication cost against accuracy: a larger period $\tau$ reduces aggregation rounds (and thus the privacy budget spent on communication) at the expense of more drift between devices.

$$
\begin{aligned}
w_m^k &= \theta_m^{k-1} - \eta\bigl(g(\theta_m^{k-1}) + b_m^k\bigr), \qquad b_m^k \sim \mathcal{N}(0, \sigma_m^2 I_d), \\
\theta_m^k &= \begin{cases} \dfrac{1}{M}\sum_{m \in \mathcal{M}} w_m^k, & k \bmod \tau = 0, \\ w_m^k, & \text{otherwise}, \end{cases}
\end{aligned}
$$

where $\theta_m^k$ is the local model on device $m$ at iteration $k$, $\eta$ is the learning rate, $g(\theta) = \frac{1}{|\mathcal{X}_m|}\sum_{n \in \mathcal{X}_m} \nabla \ell(\theta; x_n^m, y_n^m)$ is the mini-batch stochastic gradient, $b_m^k$ is per-step Gaussian noise with standard deviation $\sigma_m$ (set from the privacy requirement and gradient sensitivity), $\mathcal{M} = [1, \dots, M]$ is the set of devices, and $\tau$ is the global aggregation period.

Reference: Rui Hu, Yuanxiong Guo, E. Paul Ratazzi, Yanmin Gong, "Differentially Private Federated Learning for Resource-Constrained Internet of Things", 2020. https://arxiv.org/abs/2003.12705

---
[Back to the Canon](../README.md)
