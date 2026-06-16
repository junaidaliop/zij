# Adagrad

Implements Adagrad, a subgradient method that adapts a per-coordinate
learning rate from the history of squared gradients.

Adagrad accumulates the sum of squared gradients independently for each
parameter and divides the learning rate by the square root of that running
sum. Coordinates with large or frequent gradients receive small effective
steps, while rarely updated coordinates retain large steps, which suits
sparse data. The accumulator $G_t$ grows monotonically, so the effective
step size decreases over the course of training.

$$
\begin{aligned}
G_t &= G_{t-1} + g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon}\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the
gradient, $G_t$ is the running sum of squared gradients (computed per
coordinate), and $\epsilon$ is a small constant for numerical stability.

Reference: John Duchi, Elad Hazan, Yoram Singer, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization", JMLR 2011.
https://jmlr.org/papers/v12/duchi11a.html

---
[Back to the Canon](../index.md)
