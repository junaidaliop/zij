# KOALA++

Implements KOALA++, a Kalman-based optimizer that propagates a gradient-covariance product to capture directional uncertainty.

KOALA treats the network parameters $\theta$ as the latent state of a state-space model and updates them with an extended Kalman filter, collapsing the parameter covariance to a scalar $\sigma_t^2 I$ for tractability. KOALA++ lifts this isotropic assumption: instead of tracking a scalar variance it maintains a row vector $v_t \approx g_t P_{t-1}$, the product of the gradient with the parameter covariance. This vector records how uncertainty is distributed across directions, and it is propagated recursively from one step to the next without ever forming the full covariance matrix, so cost stays close to first-order methods.

At each step the gradient $g_t$ is correlated against the previous gradient to form a recursion for $v_t$, which is then used as a Kalman gain to scale the parameter update by the current loss residual:

$$
\begin{aligned}
\alpha_t &= \frac{g_t g_{t-1}^{\top}}{\lVert g_{t-1} \rVert^2}, \qquad
\lambda_t = \frac{g_t\!\left(v_{t-1}^{\top} + Q\, g_{t-1}^{\top}\right)}{S_{t-1}}, \\
w_t &= \frac{\left(g_t v_{t-1}^{\top}\right)\!\left(g_{t-1} g_{t-1}^{\top}\right) - \left(g_{t-1} v_{t-1}^{\top}\right)\!\left(g_t g_{t-1}^{\top}\right)}{\lVert g_{t-1} \rVert^4}, \\
v_t &= (\alpha_t - \lambda_t)\, v_{t-1} + (g_t - \lambda_t\, g_{t-1})\, Q + w_t\, g_{t-1}, \\
\theta_t &= \theta_{t-1} - \frac{\eta_t\, L_t(\theta_{t-1})}{g_t v_t^{\top} + g_t Q\, g_t^{\top} + R}\,\left(v_t^{\top} + Q\, g_t^{\top}\right).
\end{aligned}
$$

where $L_t(\theta_{t-1})$ is the minibatch loss residual, $g_t = \nabla L_t(\theta_{t-1})$ the gradient row vector, $v_t$ the gradient-covariance product surrogate (initialized $v_1 = \sigma_0^2 g_1$), $S_{t-1}$ the prior innovation covariance, $Q$ and $R$ the process and observation noise hyperparameters, and $\eta_t$ the learning rate. Setting $w_t = 0$ gives the cheaper asymmetric variant.

Reference: Zixuan Xia, Aram Davtyan, Paolo Favaro, "KOALA++: Efficient Kalman-Based Optimization of Neural Networks with Gradient-Covariance Products", arXiv 2025. https://arxiv.org/abs/2506.04432

---
[Back to the Canon](../index.md)
