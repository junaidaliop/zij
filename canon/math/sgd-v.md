# μ²-SGD

Implements μ²-SGD, a stable stochastic optimizer built from a double momentum mechanism.

μ²-SGD couples two complementary momentum ideas. The query points $x_t$ follow an Anytime-style weighted average of the iterates $w_t$, while the gradient estimate $d_t$ is a STORM-style variance-reduced (corrected) momentum that reuses the previous query point under the freshly drawn sample. With importance weights $\alpha_t = t+1$ and decay $\beta_t = 1/\alpha_t$, the estimation error shrinks as $\mathcal{O}(1/t)$, allowing a large, near-constant effective step.

Each step draws a sample $z_{t+1}$, evaluates the gradient at both the new query point $x_{t+1}$ and the previous query point $x_t$, and combines them:

$$
\begin{aligned}
g_{t+1} &= \nabla f(x_{t+1}; z_{t+1}), \qquad \bar{g}_t = \nabla f(x_t; z_{t+1}), \\
d_{t+1} &= g_{t+1} + (1 - \beta_{t+1})\,(d_t - \bar{g}_t), \\
w_{t+1} &= \Pi_{\mathcal{K}}\!\left(w_t - \eta\,\alpha_t\,d_t\right), \\
x_{t+1} &= \frac{\alpha_{1:t}}{\alpha_{1:t+1}}\,x_t + \frac{\alpha_{t+1}}{\alpha_{1:t+1}}\,w_{t+1}.
\end{aligned}
$$

where $\theta$ is identified with the iterate $w_t$, $x_t$ is the averaged query point, $d_t$ the corrected momentum gradient estimate, $\eta$ the learning rate, $\alpha_t = t+1$ the importance weights with $\alpha_{1:t} = \sum_{\tau=1}^{t}\alpha_\tau$, $\beta_t = 1/\alpha_t$ the momentum decay, $\Pi_{\mathcal{K}}$ the projection onto the feasible set, and $\bar{g}_t$ the gradient at the old query point under the new sample.

Reference: Kfir Y. Levy, "μ²-SGD: Stable Stochastic Optimization via a Double Momentum Mechanism", arXiv 2023. https://arxiv.org/abs/2304.04172

---
[Back to the Canon](../README.md)
