# pbSGD

Implements pbSGD, stochastic gradient descent with the Powerball transform applied elementwise to the gradient.

pbSGD raises each gradient component to a fixed power $\gamma \in (0, 1]$ while preserving its sign, a nonlinear reshaping called the Powerball function. Powers below one amplify small-magnitude gradients and compress large ones, which speeds up early training and improves robustness to vanishing gradients; at $\gamma = 1$ the method reduces to ordinary SGD. The momentum variant pbSGDM accumulates the transformed gradient in a velocity buffer before the step.

$$
\begin{aligned}
\sigma_\gamma(g_t) &= \mathrm{sign}(g_t)\,\lvert g_t \rvert^{\gamma} \\
m_t &= \beta\, m_{t-1} + \sigma_\gamma(g_t) \\
\theta_{t+1} &= \theta_t - \eta\, m_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $\gamma \in (0,1]$ the power exponent, $\mathrm{sign}$ and $\lvert \cdot \rvert$ act elementwise, $\beta$ the momentum factor ($\beta = 0$ recovers plain pbSGD, $\beta > 0$ gives pbSGDM), and $m_t$ the momentum buffer.

Reference: Beitong Zhou, Jun Liu, Weigao Sun, Ruijuan Chen, Claire Tomlin, Ye Yuan, "pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization", IJCAI 2020. https://www.ijcai.org/proceedings/2020/451

---
[Back to the Canon](../README.md)
