# Anon

Implements Anon, a unified adaptive optimizer with a tunable adaptivity exponent that interpolates between SGD and Adam and extrapolates beyond both.

Anon keeps the usual first and second moments, but raises the second moment to a tunable power $\gamma$ before forming the preconditioner, so $\gamma$ continuously controls how adaptive the step is ($\gamma \approx 0$ recovers SGD-like behavior, $\gamma \approx 1$ recovers Adam-like behavior). The preconditioner is refreshed only at logarithmically spaced steps through an Infrequent Decoupled Update: the accumulated second moment is collapsed into a new preconditioner via a harmonic mean with the previous one, then the accumulator is reset. Between refreshes the same preconditioner is reused, which decouples the adaptation cadence from the per-step update and stabilizes the geometry.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
s_t &= \beta_2 s_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_2^{t}} \\
\text{if } k+1 = \log_2 t:\quad
\sigma_k &= \frac{s_t}{1-\beta_2^{\max(t/2,\,1)}} + \epsilon \\
v_k &= \begin{cases} \sqrt{\dfrac{2}{\,v_{k-1}^{-2} + \sigma_k^{\gamma}\,}} & k>0 \\ \sigma_k^{-\gamma/2} & k=0 \end{cases},\qquad s_t \leftarrow 0,\quad k \leftarrow k+1 \\
\theta_t &= \Pi_{\mathcal{F},\,V_k^{-1}}\!\left(\theta_{t-1} - \eta(t)\, V_k\, \hat{m}_t\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta(t)$ the (possibly scheduled) learning rate, $g_t$ the gradient, $m_t$ and $s_t$ the first and second moments with decays $\beta_1,\beta_2$, $\epsilon$ a stability constant, $\gamma$ the adaptivity exponent, $v_k$ the harmonic-mean preconditioner refreshed only when $k+1=\log_2 t$, $V_k=\mathrm{diag}(v_k)$, and $\Pi_{\mathcal{F},V_k^{-1}}$ the projection onto the feasible set $\mathcal{F}$ in the $V_k^{-1}$ metric.

Reference: Yiheng Zhang, Kaiyan Zhao, Shaowu Wu, Yiming Wang, Jiajun Wu, Leong Hou U, Steve Drew, Xiaoguang Niu, "Anon: Extrapolating Adaptivity Beyond SGD and Adam", ICML 2025. https://arxiv.org/abs/2605.02317

---
[Back to the Canon](../index.md)
