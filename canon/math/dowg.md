# DoWG

Implements DoWG (Distance over Weighted Gradients), a parameter-free gradient method that needs no learning-rate tuning.

DoWG keeps a running estimate of the distance from the initial point and accumulates a distance-weighted sum of squared gradient norms. The step size is formed as the ratio of the squared distance estimate to the square root of this weighted sum, so no learning rate is supplied by the user; only a small initial distance estimate $r_\epsilon$ is required.

$$
\begin{aligned}
\bar{r}_t &= \max\left(\lVert \theta_t - \theta_0 \rVert,\ \bar{r}_{t-1}\right) \\
v_t &= v_{t-1} + \bar{r}_t^{\,2} \lVert g_t \rVert^2 \\
\eta_t &= \frac{\bar{r}_t^{\,2}}{\sqrt{v_t}} \\
\theta_{t+1} &= \theta_t - \eta_t\, g_t
\end{aligned}
$$

where $\theta_0$ is the initial point, $g_t = \nabla f(\theta_t)$ is the gradient, $\bar{r}_t$ is the running distance estimate initialized at $\bar{r}_{-1} = r_\epsilon > 0$, $v_t$ is the distance-weighted squared-gradient sum with $v_{-1} = 0$, and $\eta_t$ is the resulting adaptive step size.

Reference: Ahmed Khaled, Konstantin Mishchenko, Chi Jin, "DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method", NeurIPS 2023. https://arxiv.org/abs/2305.16284

---
[Back to the Canon](../README.md)
