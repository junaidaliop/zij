# ZO-AdaMM

Implements ZO-AdaMM, a zeroth-order (gradient-free) variant of AMSGrad for black-box optimization.

ZO-AdaMM replaces the true gradient with a one-point random-direction estimate computed from function-value queries alone, then feeds it into an AMSGrad-style adaptive-momentum update. The key technical point is that the parameter step is followed by a Mahalanobis-distance projection onto the feasible set $\mathcal{X}$ using the adaptive matrix $\sqrt{\hat{V}_t}$; this matched projection is what makes the method provably converge, unlike a naive Euclidean projection.

$$
\begin{aligned}
\hat{g}_t &= \frac{d}{\mu}\big[f(\theta_t + \mu u_t) - f(\theta_t)\big]\,u_t \\
m_t &= \beta_{1,t}\, m_{t-1} + (1 - \beta_{1,t})\,\hat{g}_t \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\,\hat{g}_t^2 \\
\hat{v}_t &= \max(\hat{v}_{t-1}, v_t), \qquad \hat{V}_t = \mathrm{diag}(\hat{v}_t) \\
\theta_{t+1} &= \Pi_{\mathcal{X},\,\sqrt{\hat{V}_t}}\!\Big(\theta_t - \alpha_t\, \hat{V}_t^{-1/2}\, m_t\Big)
\end{aligned}
$$

where $\hat{g}_t$ is the zeroth-order gradient estimate, $u_t$ a random vector drawn uniformly from the unit sphere, $d$ the problem dimension, $\mu$ the smoothing parameter, $\alpha_t$ the step size, $\beta_{1,t},\beta_2 \in (0,1]$ the momentum decay rates, $\hat{v}_t$ the running elementwise maximum of the second moment, and $\Pi_{\mathcal{X},H}(a) = \arg\min_{\theta \in \mathcal{X}} \|H(\theta - a)\|_2^2$ the Mahalanobis projection onto $\mathcal{X}$.

Reference: Xiangyi Chen, Sijia Liu, Kaidi Xu, Xingguo Li, Xue Lin, Mingyi Hong, David Cox, "ZO-AdaMM: Zeroth-Order Adaptive Momentum Method for Black-Box Optimization", NeurIPS 2019. https://arxiv.org/abs/1910.06513

---
[Back to the Canon](../index.md)
