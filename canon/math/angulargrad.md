# AngularGrad

Implements AngularGrad, an Adam variant that scales the first moment by the angle between consecutive gradients.

AngularGrad augments Adam with the angular behavior of the gradient trajectory. At each step it forms the tangent of the angle between the current and previous gradients, tracks the smallest such angle seen so far, and turns it into an angular coefficient $\phi_t \in [0.5, 1]$ via a $\tanh$ squashing. This coefficient rescales the first moment before the usual Adam update, so directions with small angular change between successive gradients take fuller steps while abrupt changes are damped. Two variants are defined: a tangent variant using $\tan\theta_t$ and a cosine variant using $\cos\theta_t$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\tan\theta_t &= \left| \frac{g_{t-1} - g_t}{1 + g_{t-1} g_t} \right|, \qquad \cos\theta_t = \frac{1}{\sqrt{1 + \tan^2\theta_t}} \\
\phi_t &= \tanh\!\left( \left| \xi_t \right| \right) \cdot 0.5 + 0.5, \qquad \xi_t \in \{ \tan\theta_t,\; \cos\theta_t \} \\
\theta_t &= \theta_{t-1} - \frac{\gamma \sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \cdot \frac{\phi_t\, m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $g_t$ is the current gradient and $g_{t-1}$ the previous one, $m_t, v_t$ are the first and second moment estimates, $\beta_1, \beta_2$ their decay rates, $\theta_t$ the angle between consecutive gradients, $\xi_t$ the angular term ($\tan\theta_t$ for the tangent variant, $\cos\theta_t$ for the cosine variant), $\phi_t$ the resulting angular coefficient, $\gamma$ the learning rate, and $\epsilon$ the stability constant. The coefficient $\phi_t$ uses the angular term recorded at the iteration achieving the minimum angle so far.

Reference: S. K. Roy, M. E. Paoletti, J. M. Haut, S. R. Dubey, P. Kar, A. Plaza, B. B. Chaudhuri, "AngularGrad: A New Optimization Technique for Angular Convergence of Convolutional Neural Networks", arXiv 2021. https://arxiv.org/abs/2105.10190

---
[Back to the Canon](../README.md)
