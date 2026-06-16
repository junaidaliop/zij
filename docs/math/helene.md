# HELENE

Implements HELENE, a zeroth-order fine-tuning method combining a diagonal Hessian preconditioner with layer-wise clipping and gradient annealing.

HELENE estimates the gradient with simultaneous perturbation stochastic approximation (SPSA), avoiding any backward pass. It scales each coordinate by an EMA of a diagonal Hessian estimate (an asymptotic Gauss-Newton-Bartlett estimator refreshed every $k$ steps), clipped per layer to bound the conditioning, and anneals the gradient contribution to the momentum over training to stabilize early steps.

$$
\begin{aligned}
g_t &= \frac{\mathcal{L}(\theta_t + \epsilon z;\, \mathcal{B}) - \mathcal{L}(\theta_t - \epsilon z;\, \mathcal{B})}{2\epsilon}\, z, \quad z \sim \mathcal{N}(0, I) \\
\alpha_t &= \beta_1 + (1-\beta_1)\exp(-t/T) \\
m_t &= \beta_1 m_{t-1} + \alpha_t\, g_t \\
\hat{h}_t &= \frac{1}{B}\sum_{b=1}^{B} \nabla_\theta \mathcal{L}_b \odot \nabla_\theta \mathcal{L}_b \\
h_t &= \beta_2\, h_{t-k} + (1-\beta_2)\, \hat{h}_t \\
\theta_{t+1,i} &= \theta_{t,i} - \eta_t \cdot \frac{m_{t,i}}{\gamma \cdot \max(h_{t,i},\, \lambda_i) + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t$ the SPSA gradient estimate, $\epsilon$ the perturbation scale (and the stability constant in the denominator), $z$ a standard Gaussian perturbation, $\mathcal{B}$ a minibatch, $\alpha_t$ the annealing coefficient with horizon $T$, $m_t$ the annealed gradient EMA, $\hat{h}_t$ the asymptotic Gauss-Newton-Bartlett diagonal Hessian estimate over a batch of size $B$ (refreshed every $k$ steps), $h_t$ its EMA, $\beta_1,\beta_2$ the decay rates, $\gamma$ a scaling coefficient, and $\lambda_i$ the per-layer clipping threshold applied to layer $i$.

Reference: Huaqin Zhao, Jiaxi Li, Yi Pan, Shizhe Liang, Xiaofeng Yang, Wei Liu, Xiang Li, Fei Dou, Tianming Liu, Jin Lu, "HELENE: Hessian Layer-wise Clipping and Gradient Annealing for Accelerating Fine-Tuning LLM with Zeroth-Order Optimization", arXiv 2024. https://arxiv.org/abs/2411.10696

---
[Back to the Canon](../index.md)
