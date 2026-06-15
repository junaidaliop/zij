# HGM

Implements HGM (Hindsight-Guided Momentum), an Adam variant that modulates the learning rate by the agreement between the current gradient and accumulated momentum.

HGM keeps Adam's first and second moments but adds a "hindsight" signal: the cosine similarity between the current gradient $g_t$ and the previous momentum $m_{t-1}$. When the gradient aligns with momentum the optimizer is on a consistent descent direction and the step is amplified; when they disagree the step is dampened. The similarity is smoothed over time and mapped to a multiplicative scale on the base learning rate through an exponential.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
c_t &= \frac{g_t \cdot m_{t-1}}{\lVert g_t \rVert \, \lVert m_{t-1} \rVert + \epsilon} \\
s_t &= \beta_s s_{t-1} + (1-\beta_s) c_t \\
\eta_t &= \alpha \exp(\gamma s_t) \\
\theta_t &= \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha$ the base learning rate, $g_t$ the gradient, $m_t, v_t$ the first and second moment estimates with decays $\beta_1, \beta_2$, $c_t$ the cosine similarity between the gradient and the previous momentum, $s_t$ its exponential moving average with smoothing coefficient $\beta_s$, $\gamma$ the modulation strength scaling the effective learning rate $\eta_t$, and $\epsilon$ a stability constant.

Reference: Krisanu Sarkar, "Hindsight-Guided Momentum (HGM) Optimizer: An Approach to Adaptive Learning Rates", arXiv preprint 2025. https://arxiv.org/abs/2506.22479

---
[Back to the Canon](../README.md)
