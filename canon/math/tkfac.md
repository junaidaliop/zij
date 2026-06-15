# TKFAC

Implements TKFAC, a trace-restricted Kronecker-factored approximation to the natural gradient.

Like KFAC, TKFAC approximates the Fisher information matrix as block-diagonal across layers and decomposes each block $F_l$ into a Kronecker product of two small factors. The difference is the decomposition: TKFAC writes $F_l = \delta_l\,\Phi_l \otimes \Psi_l$ and chooses the factors and the scalar $\delta_l$ via a quadratic-form estimator so that the trace of the approximate block matches the trace of the exact one. The factors are built from $\Lambda_{l-1} = a_{l-1} a_{l-1}^\top$, the covariance of the layer input activations, and $\Gamma_l = g_l g_l^\top$, the covariance of the pre-activation gradients, each weighted by the trace of the other.

For inversion the factors are damped (adding $\sqrt{\lambda}I$ after folding in $\sqrt{\delta_l}$, in the Martens-Grosse style) and tracked with an exponential moving average; the preconditioned gradient is then formed from the two factor inverses and applied with momentum. The per-layer update is:

$$
\begin{aligned}
\delta_l &= \mathbb{E}\!\left[\mathrm{tr}(\Lambda_{l-1})\,\mathrm{tr}(\Gamma_l)\right], \\
\Phi_l &= \frac{\mathbb{E}\!\left[\mathrm{tr}(\Gamma_l)\,\Lambda_{l-1}\right]}{\delta_l}, \qquad
\Psi_l = \frac{\mathbb{E}\!\left[\mathrm{tr}(\Lambda_{l-1})\,\Gamma_l\right]}{\delta_l}, \\
\hat{\Phi}_l &= \sqrt{\delta_l}\,\Phi_l + \sqrt{\lambda}\,I, \qquad
\hat{\Psi}_l = \sqrt{\delta_l}\,\Psi_l + \sqrt{\lambda}\,I, \\
\zeta_l &= -\alpha\,\big(\hat{\Phi}_l^{-1} \otimes \hat{\Psi}_l^{-1}\big)\,\nabla_{\theta_l} h, \\
m_l &\leftarrow \tau\,m_l + \zeta_l, \qquad
\theta_l \leftarrow \theta_l + m_l.
\end{aligned}
$$

where $\theta_l$ are the layer parameters, $\alpha$ the learning rate, $\nabla_{\theta_l} h$ the gradient, $\lambda$ the damping, $\tau$ the momentum coefficient, $\delta_l$ the trace-restriction scalar, $\Phi_l,\Psi_l$ the input and output Kronecker factors, $\hat{\Phi}_l,\hat{\Psi}_l$ their damped forms (which are also maintained as exponential moving averages with decay $\varepsilon$), and $\otimes$ the Kronecker product.

Reference: Kai-Xin Gao, Xiao-Lei Liu, Zheng-Hai Huang, Min Wang, Zidong Wang, Dachuan Xu, Fan Yu, "A Trace-restricted Kronecker-Factored Approximation to Natural Gradient", AAAI 2021. https://arxiv.org/abs/2011.10741

---
[Back to the Canon](../README.md)
