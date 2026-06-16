# AdaSAM

Implements AdaSAM, sharpness-aware minimization combined with an adaptive learning rate and momentum.

SAM seeks parameters in flat loss regions by computing the gradient at a worst-case perturbation $\delta_t = \rho\, g_t / \lVert g_t \rVert$ within a $\rho$-ball, but uses a fixed global step size. AdaSAM feeds the perturbed gradient through Adam-style first and second moments with an AMSGrad maximum on the second moment, giving each coordinate its own adaptive step. Numerical stability is provided by initializing $\hat{v}_{-1} = \epsilon^2$ rather than by an added constant in the denominator.

$$
\begin{aligned}
\tilde{g}_t &= \nabla f\!\left(\theta_t + \rho\,\frac{g_t}{\lVert g_t \rVert}\right), \quad g_t = \nabla f(\theta_t) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde{g}_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\tilde{g}_t^{\,2} \\
\hat{v}_t &= \max(\hat{v}_{t-1},\, v_t) \\
\theta_{t+1} &= \theta_t - \gamma\, \frac{m_t}{\sqrt{\hat{v}_t}}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the base learning rate, $\rho$ the perturbation radius, $g_t$ the gradient at $\theta_t$, $\tilde{g}_t$ the gradient at the perturbed point, $m_t$/$v_t$ the first and second moments, $\hat{v}_t$ the running coordinate-wise maximum of the second moment, $\beta_1,\beta_2$ the decay rates, $\epsilon$ the stability constant via $\hat{v}_{-1}=\epsilon^2$, and all squaring and division are element-wise.

Reference: Hao Sun, Li Shen, Qihuang Zhong, Liang Ding, Shixiang Chen, Jingwei Sun, Jing Li, Guangzhong Sun, Dacheng Tao, "AdaSAM: Boosting Sharpness-Aware Minimization with Adaptive Learning Rate and Momentum for Training Deep Neural Networks", arXiv 2023. https://arxiv.org/abs/2303.00565

---
[Back to the Canon](../index.md)
