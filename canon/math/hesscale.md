# HesScale

Implements AdaHesScale, an Adam-style optimizer that replaces the squared-gradient second moment with HesScale's diagonal Hessian approximation.

HesScale estimates the diagonal of the Hessian by backpropagating second-order information layer by layer, using exact Hessian diagonals at the output layer and a curvature-only recursion (dropping off-diagonal coupling) through the hidden layers, so the cost matches a standard gradient backward pass. AdaHesScale then preconditions the gradient with an exponential moving average of the squared diagonal Hessian estimate $h_t$ in place of $g_t^2$, giving a scalable second-order step within the familiar adaptive-moment framework.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) h_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $g_t$ is the gradient, $h_t$ is the HesScale per-parameter diagonal Hessian estimate, $m_t$ and $v_t$ are the first- and second-moment EMAs, $\beta_1,\beta_2$ are decay rates, $\eta$ is the step size, and $\epsilon$ is a stability constant.

Reference: Mohamed Elsayed, Homayoon Farrahi, Felix Dangel, A. Rupam Mahmood, "Revisiting Scalable Hessian Diagonal Approximations for Applications in Reinforcement Learning", ICML 2024. https://arxiv.org/abs/2406.03276

---
[Back to the Canon](../README.md)
