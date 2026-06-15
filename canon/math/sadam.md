# Sadam

Implements Sadam, a calibrated Adam that replaces the $\sqrt{v_t}+\epsilon$ denominator with a softplus of $\sqrt{v_t}$.

Sadam (softplus-Adam) addresses the unstable adaptive learning rate that Adam produces when the second moment $v_t$ is tiny: the factor $1/(\sqrt{v_t}+\epsilon)$ blows up and the choice of $\epsilon$ becomes delicate. The fix is to pass $\sqrt{v_t}$ through a softplus activation $\mathrm{sp}_\beta$, which is lower-bounded and grows linearly for large arguments. This smoothly clips the extreme step sizes while keeping the adaptivity of Adam, and it removes $\epsilon$ entirely.

The first and second moments are accumulated exactly as in Adam, and the per-coordinate step divides the momentum by the softplus-calibrated denominator:

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2 \\
\theta_{t+1} &= \theta_t - \eta_t\, \frac{m_t}{\mathrm{sp}_\beta\!\left(\sqrt{v_t}\right)} \\
\mathrm{sp}_\beta(x) &= \frac{1}{\beta}\log\!\left(1 + e^{\beta x}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first and second moments, $\beta_1,\beta_2$ their decay rates, and $\beta$ the softplus sharpness (larger $\beta$ recovers Adam's $\sqrt{v_t}$ denominator). All operations are element-wise. The recommended settings are $\beta_1=0.9$, $\beta_2=0.999$, $\beta=50$, and no $\epsilon$ is needed.

Reference: Qianqian Tong, Guannan Liang, Jinbo Bi, "Calibrating the Adaptive Learning Rate to Improve Convergence of ADAM", arXiv 2019. https://arxiv.org/abs/1908.00700

---
[Back to the Canon](../README.md)
