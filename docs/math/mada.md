# MADA

Implements MADA, a meta-adaptive optimizer that learns where to sit between known adaptive methods via hyper-gradient descent.

MADA defines a single parameterized update that subsumes Adam, AMSGrad, Adan, and Yogi as special points in a continuous coefficient space. The first moment blends plain momentum with Adan's gradient-difference term ($\beta_3$), and the second moment blends an Adam-style running average, a Yogi-style sign correction ($c$), and an AMSGrad-style running maximum ($\rho$). These interpolation coefficients are not fixed: they are treated as additional variables and updated during training by descending the validation/training loss with respect to them (hyper-gradient descent), so the optimizer drifts toward whichever known method works best for the task. Bias-correction terms are omitted below for clarity, as in the paper.

$$
\begin{aligned}
\bar{m}_t &= \beta_1 \bar{m}_{t-1} + (1-\beta_1) g_t \\
n_t &= \beta_3 n_{t-1} + (1-\beta_3)(g_t - g_{t-1}) \\
m_t &= \bar{m}_t + \beta_3 n_t \\
\hat{g}_t &= g_t + \beta_3 (g_t - g_{t-1}) \\
\tilde{g}_t^2 &= c\,\hat{g}_t^2 + (1-c)\left(v_{t-1} + \hat{g}_t^2 \cdot \mathrm{sign}(\hat{g}_t^2 - v_{t-1})\right) \\
\tilde{v}_t &= \beta_2 \tilde{v}_{t-1} + (1-\beta_2)\,\tilde{g}_t^2 \\
v_t^{\max} &= \max\left(v_{t-1}^{\max},\, \tilde{v}_t\right) \\
v_t &= \rho\,\tilde{v}_t + (1-\rho)\,v_t^{\max} \\
\theta_t &= \theta_{t-1} - \eta_t \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first and second moments, $\beta_1,\beta_2$ the moment decays, $\epsilon$ the stability constant, and $\beta_3, c, \rho$ the interpolation coefficients (learned by hyper-gradient descent): $\beta_3$ weights Adan's gradient-difference term, $c$ interpolates Yogi's sign correction, and $\rho$ interpolates the AMSGrad running maximum. Setting $\beta_3=0$, $c=1$, $\rho=1$ recovers Adam.

Reference: Kaan Ozkara, Can Karakus, Parameswaran Raman, Mingyi Hong, Shoham Sabach, Branislav Kveton, Volkan Cevher, "MADA: Meta-Adaptive Optimizers through hyper-gradient Descent", arXiv 2024. https://arxiv.org/abs/2401.08893

---
[Back to the Canon](../index.md)
