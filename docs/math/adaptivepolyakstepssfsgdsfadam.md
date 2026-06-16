# Adaptive Polyak Steps (SF-SGD / SF-Adam)

Implements Adaptive Polyak Steps (SF-SGD / SF-Adam), Polyak-type step sizes that make Schedule-Free SGD and Adam fully learning-rate-free.

Schedule-Free SGD maintains three sequences: a gradient-evaluation point $y_t$, a base point $z_t$ where the gradient step lands, and the returned iterate average $\theta_t$. Rather than tuning a fixed base step, the method sets $\gamma_t$ at each iteration from the sampled loss, the gradient, and the current iterates by minimizing an upper bound on the distance to the solution. This extends the $\mathrm{SPS}_+$ rule to the schedule-free averaging scheme; setting $\beta=0$ (so $y_t=z_{t-1}$) recovers the standard SGD Polyak rule.

Two step sizes are given: an oracle form using the per-sample optimal loss $f_{\zeta_t}(\theta_\star)$, and a safeguarded form that replaces it with any lower bound $\ell_{\zeta_t}^\star$ and caps the denominator with $M$ to prevent blow-up. The Adam variant keeps the same updates but measures the gradient in the norm induced by the inverse of the diagonal Adam preconditioner $D_t$, and steps with $D_t^{-1}$.

$$
\begin{aligned}
y_t &= (1-\beta)\,z_{t-1} + \beta\,\theta_t \\
v_t &= \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2, \qquad g_t = \nabla f_{\zeta_t}(y_t) \\
D_t &= \mathrm{diag}\!\left(\sqrt{v_t/(1-\beta_2^{\,t+1})} + \epsilon\right) \\
\gamma_t &= \frac{\left[\,f_{\zeta_t}(y_t) - \ell_{\zeta_t}^\star + \beta\,\langle g_t,\, z_{t-1}-\theta_t\rangle\,\right]_+}{\max\!\left\{\,\lVert g_t\rVert_{D_t^{-1}}^2,\; M\,\right\}} \\
z_t &= z_{t-1} - \gamma_t\, D_t^{-1} g_t \\
\theta_{t+1} &= (1-c_{t+1})\,\theta_t + c_{t+1}\, z_t, \qquad c_{t+1} = \frac{1}{t+1}
\end{aligned}
$$

where $\theta_t$ is the returned iterate average, $z_t$ the base sequence ($z_{-1}=\theta_0$), $y_t$ the gradient-evaluation point, $\beta\in[0,1)$ the schedule-free momentum, $g_t$ the sampled gradient, $v_t$ the second-moment estimate with decay $\beta_2$ and stability $\epsilon$, $D_t$ the diagonal Adam preconditioner, $\lVert v\rVert_{D_t^{-1}}^2 = v^\top D_t^{-1} v$, $c_{t+1}$ the averaging weight, $[\,\cdot\,]_+=\max\{\cdot,0\}$, $\ell_{\zeta_t}^\star \le f_{\zeta_t}(\theta_\star)$ a lower bound on the sampled loss, and $M>0$ the safeguard. The SGD variant is the special case $D_t=I$; the oracle step replaces $\ell_{\zeta_t}^\star$ by $f_{\zeta_t}(\theta_\star)$ and drops the $M$ safeguard from the denominator.

Reference: Dimitris Oikonomou, Matthew Buchholz, Yuen-Man Pun, Robert M. Gower, Nicolas Loizou, "Taking the Road Less Scheduled with Adaptive Polyak Steps", arXiv 2025. https://arxiv.org/abs/2511.07767

---
[Back to the Canon](../index.md)
