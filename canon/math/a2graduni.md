# A2GradUni

Implements A2Grad (uniform variant), adaptive accelerated SGD.

Accelerated stochastic gradient descent with a diagonal adaptive term.
Three sequences are coupled per step: a gradient-evaluation point
$x_k$, an averaged point, and the iterate $\theta_k$. The
step coefficient mixes the Lipschitz term $\gamma_k$ with an
adaptive accumulation $h_k$ of the gradient deviation from its
running average:


$$
\begin{aligned}
\gamma_k &= \frac{2 L}{k + 1} \\
\bar{g}_k &= \frac{1}{k + 1} \sum_{i=0}^{k} g_i \\
\delta_k &= g_k - \bar{g}_k \\
v_k &= v_{k-1} + \lVert \delta_k \rVert^2,
     \qquad h_k = \sqrt{v_k} \\
\alpha_k &= \frac{2}{k + 3},
     \qquad c_k = \frac{1}{\gamma_k + \beta h_k} \\
x_{k+1} &= x_k - c_k\, g_k \\
\theta_{k+1} &= (1 - \alpha_k)\,\theta_k + \alpha_k\, x_{k+1}
                - (1 - \alpha_k)\,\alpha_{k-1}\, c_k\, g_k
\end{aligned}
$$

Reference: Qi Deng, Yi Cheng, Guanghui Lan, "Optimal Adaptive and
Accelerated Stochastic Gradient Descent", arXiv 2018.
https://arxiv.org/abs/1810.00553

---
[Back to the Canon](../README.md)
