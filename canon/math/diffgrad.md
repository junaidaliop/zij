# DiffGrad

Implements DiffGrad, Adam scaled by a gradient-change friction coefficient.

The friction coefficient $\xi_t \in [0.5, 1]$ is the sigmoid of the
absolute change between consecutive gradients, so the first moment is damped
toward half its value where consecutive gradients agree (the gradient is
stable) and passed through nearly unchanged where they differ sharply (the
gradient is changing quickly):


$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\xi_t &= \frac{1}{1 + e^{-|g_{t-1} - g_t|}} \\
\theta_t &= \theta_{t-1} - \eta\,
            \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}\,
            \frac{\xi_t\, m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

Reference: Shiv Ram Dubey, Soumendu Chakraborty, Swalpa Kumar Roy,
Snehasis Mukherjee, Satish Kumar Singh, Bidyut Baran Chaudhuri,
"diffGrad: An Optimization Method for Convolutional Neural Networks",
IEEE Transactions on Neural Networks and Learning Systems 2020.
https://arxiv.org/abs/1909.11015

---
[Back to the Canon](../README.md)
