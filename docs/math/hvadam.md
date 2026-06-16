# HVAdam

Implements HVAdam, an Adam variant that preconditions with a hidden vector of invariant gradient components.

HVAdam targets the "valley dilemma," where the descent direction along a valley floor carries small, persistent gradients that per-dimension adaptive methods suppress as noise. It maintains a hidden vector $v_t$ tracking the stable gradient trend, measures the per-dimension squared deviation of the current gradient from that trend, and uses the relative size of this deviation to scale the second-moment estimate. A hidden-vector term, gated by an adaptive step $b_t$ derived from the running cosine similarity between $v_t$ and the bias-corrected momentum (with a restart when alignment decays), is added to the standard adaptive step.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
p_t &= (g_t - v_{t-1})^2 \\
\eta_t &= \frac{p_t}{(g_t - m_t)^2 + \gamma\, p_t + \epsilon} \\
s_t &= \beta_2 s_{t-1} + (1 - \beta_2)\, \eta_t\, p_t + \epsilon \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{s}_t = \frac{s_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha_1 \frac{\hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon} - \alpha_2\, b_t\, v_t
\end{aligned}
$$

where $g_t$ is the gradient, $m_t$ the first moment, $v_t$ the hidden vector approximating the invariant gradient direction, $p_t$ the squared deviation of the gradient from $v_{t-1}$, $\eta_t$ the relative noise magnitude per dimension, $s_t$ the resulting second-moment preconditioner, $\alpha_1,\alpha_2$ the base learning rates for the adaptive and hidden-vector terms, $b_t$ the adaptive step size gating the hidden-vector contribution, $\beta_1,\beta_2$ the decay rates, $\gamma$ a constant bounding $\eta_t$, and $\epsilon$ a stability constant.

Reference: Yiheng Zhang, Shaowu Wu, Yuanzhuo Xu, Jiajun Wu, Shang Xu, Steve Drew, Xiaoguang Niu, "HVAdam: A Full-Dimension Adaptive Optimizer", arXiv 2025. https://arxiv.org/abs/2511.20277

---
[Back to the Canon](../index.md)
