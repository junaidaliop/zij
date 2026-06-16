# Natural Gradient

Implements Natural Gradient descent, steepest descent in a Riemannian parameter space preconditioned by the Fisher information matrix.

When the parameter space carries an underlying geometric structure, the ordinary gradient is not the direction of steepest descent measured in that geometry. Amari shows that the steepest-descent direction with respect to the Riemannian metric $G(\theta)$ — taken to be the Fisher information matrix of the statistical model — is the *natural gradient* $\tilde{\nabla}L(\theta) = G^{-1}(\theta)\nabla L(\theta)$. Following it makes the update invariant to reparameterization, and the resulting online estimator is asymptotically Fisher efficient.

Each step preconditions the ordinary gradient by the inverse metric before moving:

$$
\begin{aligned}
g_t &= \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta_t\, G^{-1}(\theta_t)\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $L$ the loss, $g_t = \nabla L(\theta_t)$ the ordinary gradient, and $G(\theta)$ the Fisher information matrix (the Riemannian metric on the parameter manifold) with inverse $G^{-1}(\theta)$. When $G$ is the identity the rule reduces to ordinary gradient descent.

Reference: Shun-ichi Amari, "Natural Gradient Works Efficiently in Learning", Neural Computation 10(2), 1998. https://doi.org/10.1162/089976698300017746

---
[Back to the Canon](../index.md)
