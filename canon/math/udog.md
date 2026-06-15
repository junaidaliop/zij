# U-DoG

Implements U-DoG, an accelerated parameter-free optimizer fusing the UniXGrad extragradient scheme with the DoG distance-over-gradients step size.

U-DoG removes the learning rate entirely by setting the step from observed quantities: the numerator is $\bar r_t$, the largest distance any iterate has traveled from the start $\theta_0$, and the denominator is $\sqrt{G_t}$, a running accumulation of weighted squared gradient norms. This is the DoG estimator "distance over gradients," which guesses the distance to the optimum online without prior knowledge of the domain diameter or the noise level.

To get an accelerated rate it runs a two-sequence extragradient (UniXGrad) loop: an extrapolated point $\theta_{t+1}$ is formed from a gradient $m_t$ taken at the averaged iterate, and the leading sequence $y_{t+1}$ then steps with a gradient $g_t$ taken at $\theta_{t+1}$. Increasing weights $\alpha_t$ emphasize later, better iterates, and the method returns the weighted average $\hat\theta_T$.

$$
\begin{aligned}
\bar r_t &= \max_{k \le t} \max\{\,\lVert y_k - \theta_0 \rVert,\ \lVert \theta_k - \theta_0 \rVert,\ r_\epsilon\,\} \\
\alpha_t &= \frac{1}{\bar r_t}\sum_{k=0}^{t}\bar r_k, \qquad \omega_t = \alpha_t\,\bar r_t \\
\eta_{x,t} &= \frac{\bar r_t}{\sqrt{G_{x,t}}}, \qquad \eta_{y,t} = \frac{\bar r_t}{\sqrt{G_{y,t}}} \\
\theta_{t+1} &= \mathrm{Proj}_{\mathcal{K}}\!\left(y_t - \alpha_t\,\eta_{x,t}\,m_t\right) \\
y_{t+1} &= \mathrm{Proj}_{\mathcal{K}}\!\left(y_t - \alpha_t\,\eta_{y,t}\,g_t\right)
\end{aligned}
$$

where $\theta$ are the parameters (the $x$ sequence), $y$ the leading extragradient sequence, $\theta_0$ the start, $g_t$ and $m_t$ stochastic gradients sampled at weighted averages of past iterates, $\bar r_t$ the running maximum distance from $\theta_0$, $r_\epsilon$ a small initial movement, $\alpha_t,\omega_t$ the increasing iterate weights, $G_{x,t},G_{y,t}$ the (ordered) accumulated weighted squared gradient norms, $\eta_{x,t},\eta_{y,t}$ the resulting tuning-free step sizes, $\mathcal{K}$ the feasible set with Euclidean projection $\mathrm{Proj}_{\mathcal{K}}$, and the output is the $\omega$-weighted average iterate $\hat\theta_T$.

Reference: Itai Kreisler, Maor Ivgi, Oliver Hinder, Yair Carmon, "Accelerated Parameter-Free Stochastic Optimization", arXiv 2024. https://arxiv.org/abs/2404.00666

---
[Back to the Canon](../README.md)
