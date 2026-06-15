# LPSGD / LPSGDM

Implements LPSGD / LPSGDM, SGD and SGD-with-momentum equipped with a cosine $\ell_p$-norm scheme.

The closed-form $\ell_p$ steepest-descent step rescales each gradient coordinate by $|g_t|^{\rho}$ with $\rho = (p-2)/(p-1)$, which interpolates between plain SGD ($p=2$, $\rho=0$) and sign-based updates ($p\to\infty$, $\rho\to1$); larger $p$ compresses directional imbalance from outlier-curvature directions. Rather than fixing $p$, the method anneals it per epoch with a cosine schedule that starts at $p_{\max}\ge 2$ and decays smoothly to $2$ by the final epoch, so training begins in a flatter, sign-like geometry and ends in the Euclidean geometry.

LPSGD applies this coordinate-wise rescaling directly to the stochastic gradient. LPSGDM first accumulates an exponential momentum buffer, rescales the momentum-smoothed gradient, and adds decoupled weight decay.

$$
\begin{aligned}
p_s &= 2 + (p_{\max}-2)\cdot \frac{1+\cos\!\left(\pi\,\frac{s-1}{S-1}\right)}{2}, \qquad \rho_s = \frac{p_s-2}{p_s-1} \\
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t \\
v_t &= \frac{m_t}{(|m_t| + \epsilon)^{\rho_s}} \\
\theta_{t+1} &= (1-\eta\lambda)\,\theta_t - \eta\, v_t
\end{aligned}
$$

where $s$ is the epoch index and $S$ the total number of epochs, $p_{\max}\ge 2$ is the maximum norm parameter, $\rho_s\in[0,1)$ is the epoch-dependent rescaling exponent, $g_t$ is the stochastic gradient, $m_t$ the momentum buffer, $v_t$ the rescaled update direction, $\eta$ the learning rate, $\beta\in[0,1)$ the momentum coefficient, $\lambda\ge 0$ the weight decay, $\epsilon>0$ a stability constant, and $|\cdot|$ acts element-wise. LPSGD is the special case $\beta=0$, $\lambda=0$ with the rescaling applied to $g_t$ directly: $\theta_{t+1} = \theta_t - \eta\, g_t/(|g_t|+\epsilon)^{\rho_s}$.

Reference: Jianhao Xu, Zhuang Yang, "Beyond $\ell_2$-norm and $\ell_\infty$-norm: A Curvature-Inspired $\ell_p$-Norm Scheme for Deep Neural Networks", arXiv 2026. https://arxiv.org/abs/2606.02078

---
[Back to the Canon](../README.md)
