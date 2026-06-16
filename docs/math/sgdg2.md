# SGD-G2

Implements SGD-G2, an SGD variant that adapts a single global learning rate from a finite-difference estimate of the local curvature along the gradient direction.

SGD-G2 is derived from a stochastic Runge-Kutta view of gradient descent. At each step it takes a tentative Euler step along the gradient, re-evaluates the mini-batch gradient at the probed point, and uses the change in gradient to estimate the optimal step size $h_t^{\mathrm{opt}}$ that a quadratic model would prescribe. This is a Hessian-free curvature probe: the inner product and squared norm of the gradient difference play the role of a directional second derivative. The working learning rate is then moved toward $h_t^{\mathrm{opt}}$ by an exponential rule, with a faster reaction when the estimate calls for a smaller step.

$$
\begin{aligned}
\tilde{g}_t &= \nabla f(\theta_t - h_t\,g_t), \\
h_t^{\mathrm{opt}} &= \begin{cases} \dfrac{2\,h_t\,\langle g_t - \tilde{g}_t,\; g_t\rangle}{\lVert g_t - \tilde{g}_t \rVert^2} & \text{if } \langle g_t - \tilde{g}_t,\; g_t\rangle > 0, \\ h_t & \text{otherwise}, \end{cases} \\
h_{t+1} &= \begin{cases} \beta\,h_t + (1-\beta)\,h_t^{\mathrm{opt}} & \text{if } h_t^{\mathrm{opt}} \ge h_t, \\ (1-\beta)\,h_t^{\mathrm{opt}} & \text{otherwise}, \end{cases} \\
\theta_{t+1} &= \theta_t - h_{t+1}\,g_t,
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the mini-batch gradient at $\theta_t$, $\tilde{g}_t$ is the mini-batch gradient re-evaluated after a tentative step $-h_t g_t$, $h_t$ is the current learning rate, $h_t^{\mathrm{opt}}$ is the curvature-optimal step from the quadratic model, and $\beta \in (0,1)$ is the smoothing coefficient controlling how quickly the learning rate follows $h_t^{\mathrm{opt}}$.

Reference: Imen Ayadi, Gabriel Turinici, "Stochastic Runge-Kutta methods and adaptive SGD-G2 stochastic gradient descent", 2020. https://arxiv.org/abs/2002.09304

---
[Back to the Canon](../index.md)
