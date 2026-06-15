# ESAM

Implements ESAM, an efficient sharpness-aware minimizer that approximates SAM at a fraction of its cost.

SAM seeks parameters in flat regions of the loss landscape by perturbing the weights toward the worst-case direction $\hat{\epsilon}=\rho\,g_t/\lVert g_t\rVert$ and then taking a step on the gradient evaluated at $\theta+\hat{\epsilon}$, which doubles the per-step cost. ESAM cuts this overhead with two strategies. Stochastic Weight Perturbation (SWP) perturbs only a random subset of parameters: each coordinate is kept with probability $\beta$ via a Bernoulli mask $m$ and rescaled by $1/\beta$ so the perturbation stays unbiased. Sharpness-sensitive Data Selection (SDS) computes the final gradient on only the subset $\mathcal{B}^{+}$ of the batch whose loss increases most under the perturbation, since those samples dominate the sharpness measure.

$$
\begin{aligned}
\hat{\epsilon} &= \frac{\rho}{\beta}\, m \odot \frac{g_t}{\lVert g_t\rVert}, \quad m_i \sim \mathrm{Bern}(\beta) \\
\mathcal{B}^{+} &= \{\, i \in \mathcal{B} : \ell(\theta+\hat{\epsilon}; x_i) - \ell(\theta; x_i) > \alpha \,\}, \quad |\mathcal{B}^{+}| = \gamma\,|\mathcal{B}| \\
g_t^{+} &= \nabla_\theta L_{\mathcal{B}^{+}}(\theta+\hat{\epsilon}) \\
\theta_{t+1} &= \theta_t - \eta\, g_t^{+}
\end{aligned}
$$

where $\rho$ is the neighborhood radius, $g_t=\nabla_\theta L_{\mathcal{B}}(\theta)$ is the batch gradient, $\beta\in(0,1]$ is the SWP keep probability, $m$ is the per-coordinate Bernoulli mask, $\odot$ is elementwise product, $\gamma\in(0,1]$ is the SDS selection ratio, $\alpha$ is the threshold induced by $\gamma$, $L_{\mathcal{B}^{+}}$ is the loss over the selected subset, and $\eta$ is the learning rate.

Reference: Jiawei Du, Hanshu Yan, Jiashi Feng, Joey Tianyi Zhou, Liangli Zhen, Rick Siow Mong Goh, Vincent Y. F. Tan, "Efficient Sharpness-aware Minimization for Improved Training of Neural Networks", ICLR 2022. https://arxiv.org/abs/2110.03141

---
[Back to the Canon](../README.md)
