# Gradient Centralization (GC)

Implements Gradient Centralization (GC), a gradient projection that removes the per-weight-vector mean before the optimizer step.

GC operates on the gradient of each weight vector (for example each output unit's incoming weights, or each convolution filter). It subtracts the mean of the gradient's components, which is equivalent to projecting the gradient onto the hyperplane whose normal is the all-ones direction. This centralized gradient then feeds any base optimizer (SGD with momentum, Adam, etc.) in place of the raw gradient, regularizing the solution space and smoothing training. For a weight matrix $W$ whose columns are the $M$-dimensional weight vectors, GC applies a fixed projection $P$ to each column of the gradient.

$$
\begin{aligned}
\Phi_{\mathrm{GC}}(g_t) &= g_t - \frac{1}{M}\sum_{j=1}^{M} (g_t)_j = P\, g_t, \quad P = I - e e^{\top}, \quad e = \tfrac{1}{\sqrt{M}}\mathbf{1} \\
\hat{g}_t &= \Phi_{\mathrm{GC}}(g_t) \\
m_t &= \beta\, m_{t-1} + (1-\beta)\,\hat{g}_t \\
\theta_{t+1} &= \theta_t - \gamma\, m_t
\end{aligned}
$$

where $g_t = \nabla_{\theta_t}\mathcal{L}$ is the gradient of a single weight vector with $M$ elements, $\Phi_{\mathrm{GC}}$ centralizes it (mean form, equal to the projection $P g_t$ with $I$ the $M\times M$ identity and $\mathbf{1}$ the all-ones vector), $m_t$ is the momentum buffer with decay $\beta$, $\gamma$ the learning rate, and $\theta$ the parameters. For Adam, $\hat{g}_t$ replaces the raw gradient in both the first moment $m_t$ and second moment $v_t$ before the usual bias correction and update.

Reference: Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang, "Gradient Centralization: A New Optimization Technique for Deep Neural Networks", ECCV 2020. https://arxiv.org/abs/2004.01461

---
[Back to the Canon](../README.md)
