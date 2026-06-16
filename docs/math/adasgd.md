# AdaSGD

Implements AdaSGD, an SGD-with-momentum variant that adapts a single global learning rate using Adam's second-moment estimate.

AdaSGD keeps SGD's per-coordinate update direction but borrows Adam's adaptive step size in scalar form. Instead of dividing each coordinate by its own running second moment, it tracks one scalar second moment $v_t$, the bias-corrected exponential average of the squared gradient norm, and uses it to scale a global learning rate shared by all parameters. Normalizing by the dimension $d$ keeps the scale comparable across problem sizes, so a single base rate $\eta$ transfers between tasks with little tuning while retaining SGD's implicit regularization.

$$
\begin{aligned}
m_t &= \beta_1\,m_{t-1} + g_t, \\
v_t &= \beta_2\,v_{t-1} + (1-\beta_2)\,\lVert g_t \rVert_2^2, \\
\eta_t &= \eta\,\frac{\sqrt{1-\beta_2^{\,t}}}{\sqrt{v_t/d}}, \\
\theta_{t+1} &= \theta_t - \eta_t\,m_t,
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the gradient, $m_t$ is the (unnormalized) momentum buffer with $m_0=0$, $v_t$ is the scalar second-moment estimate with $v_0=0$, $\beta_1$ and $\beta_2$ are the momentum and second-moment decay rates, $d$ is the parameter dimensionality, $\eta$ is the base learning rate, and $\eta_t$ is the resulting global step size with $\sqrt{1-\beta_2^{\,t}}$ correcting the zero initialization of $v_t$.

Reference: Jiaxuan Wang, Jenna Wiens, "AdaSGD: Bridging the gap between SGD and Adam", 2020. https://arxiv.org/abs/2006.16541

---
[Back to the Canon](../index.md)
