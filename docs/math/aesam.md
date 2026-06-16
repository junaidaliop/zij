# AE-SAM

Implements AE-SAM, an adaptive policy that applies Sharpness-Aware Minimization only on the steps that need it.

SAM improves generalization by minimizing a perturbed loss, but it doubles the cost per step because each update needs two gradients: one to build the perturbation and one to update the weights. AE-SAM tracks the running mean and variance of the squared stochastic gradient norm and triggers the full SAM step only when the current squared norm is large relative to that distribution, i.e. when the iterate sits in a sharp region; otherwise it falls back to a cheap single-gradient (ERM) step. A linearly scheduled threshold controls how often SAM fires over the course of training.

$$
\begin{aligned}
\mu_t &= \delta\,\mu_{t-1} + (1-\delta)\,\|g_t\|^2, \\
\sigma_t^2 &= \delta\,\sigma_{t-1}^2 + (1-\delta)\,\big(\|g_t\|^2 - \mu_t\big)^2, \\
c_t &= \tfrac{t}{T}\,\lambda_1 + \big(1-\tfrac{t}{T}\big)\,\lambda_2, \\
\theta_{t+1} &=
\begin{cases}
\theta_t - \eta\,\nabla\mathcal{L}\!\left(\theta_t + \dfrac{\rho\,g_t}{\|g_t\|}\right), & \|g_t\|^2 \ge \mu_t + c_t\,\sigma_t, \\
\theta_t - \eta\,g_t, & \text{otherwise},
\end{cases}
\end{aligned}
$$

where $g_t = \nabla\mathcal{L}(\theta_t)$ is the stochastic gradient on batch $\mathcal{B}_t$, $\eta$ is the learning rate, $\rho$ the SAM neighborhood radius, $\delta \in (0,1)$ the EMA forgetting rate, $\mu_t$ and $\sigma_t^2$ the running mean and variance of $\|g_t\|^2$, $T$ the total number of iterations, and $\lambda_1, \lambda_2$ the endpoints of the linear threshold schedule $c_t$.

Reference: Weisen Jiang, Hansi Yang, Yu Zhang, James Kwok, "An Adaptive Policy to Employ Sharpness-Aware Minimization", ICLR 2023. https://arxiv.org/abs/2304.14647

---
[Back to the Canon](../index.md)
