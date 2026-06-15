# FCGD_G-L

Implements FCGD_G-L, fractional calculus gradient descent built on the Grünwald–Letnikov fractional-order derivative.

The Grünwald–Letnikov definition expresses the $\alpha$-order derivative as an infinite weighted sum of past function values. FCGD_G-L applies this to the gradient: the $\alpha$-order gradient at step $t$ is the current first-order gradient plus a memory of the first-order gradients from previous steps, each scaled by a Grünwald–Letnikov binomial weight. Two simplifications make it practical. The short-memory effect truncates the history to the last 10 steps, and a recursive coefficient formula eliminates the gamma functions of the binomial term, so each weight follows from its predecessor. When $\alpha = 1$ the memory weights vanish and the rule collapses to ordinary gradient descent.

To keep the parameters from settling into a local optimum, a small disturbance is injected: a mask vector $c$ holding nine ones and a single zero is reshuffled every step, randomly dropping one of the ten memory terms. The parameter is then updated by plain descent along this fractional gradient.

$$
\begin{aligned}
w_0 &= 1, \qquad w_j = \left(1 - \frac{\alpha + 1}{j + 1}\right) w_{j-1}, \quad j = 1, 2, \dots, 10, \\
D^{\alpha}_t &= g_t + \sum_{j=1}^{10} c_j\, w_j\, g_{t-j}, \\
\theta_t &= \theta_{t-1} - \eta\, D^{\alpha}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t = \nabla f(\theta_{t-1})$ the current gradient and $g_{t-j}$ the gradients of the preceding 10 steps, $\alpha \in (0,1]$ the fractional order, $w_j$ the recursively generated Grünwald–Letnikov weights, and $c \in \{0,1\}^{10}$ a randomly permuted mask containing nine ones and one zero that drops a single memory term each step. The SGD and Adam variants FCSGD_G-L and FCAdam_G-L substitute this fractional gradient $D^{\alpha}_t$ for the plain gradient inside the respective base optimizer.

Reference: Xiaojun Zhou, Chunna Zhao, Yaqun Huang, "A Deep Learning Optimizer Based on Grünwald–Letnikov Fractional Order Definition", Mathematics 2023. https://doi.org/10.3390/math11020316

---
[Back to the Canon](../README.md)
