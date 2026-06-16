# AdaRankGrad

Implements AdaRankGrad, low-rank Adam with an adaptively shrinking gradient subspace for memory-efficient LLM training.

The gradient $g_t$ is projected onto a low-rank subspace whose rank $r_t$ is chosen at runtime from the gradient's information content: $r_t$ is the largest rank whose rank-$r$ approximation captures a fraction $\eta_{th}$ of the gradient energy, which provably decreases over training so the optimizer footprint shrinks. A projection $Q_t$ (with $r_t$ rows) maps the gradient into that subspace, where standard Adam moments are maintained, before mapping the resulting step back to the full parameter space. When the subspace changes between steps, the moments are carried over by re-expressing them in the new basis through $R_t = Q_t Q_{t-1}^\top$.

$$
\begin{aligned}
r_t &= \sup\{\, r : \eta_{th}\,\lVert g_t\rVert_F^2 - \lVert g_t - P_t(r)\,g_t\rVert_F^2 \ge 0 \,\}, \quad \hat g_t = Q_t\, g_t \\
R_t &= Q_t\, Q_{t-1}^\top, \quad m_t \leftarrow R_t\, m_{t-1}, \quad v_t \leftarrow R_t\, v_{t-1} \\
m_t &= \beta_1 m_t + (1-\beta_1)\,\hat g_t, \quad v_t = \beta_2 v_t + (1-\beta_2)\,\hat g_t^{\,2} \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \quad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \gamma\, Q_t^\top \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters (weight matrix), $\gamma$ the learning rate, $g_t$ the gradient, $Q_t$ the rank-$r_t$ projection onto the adaptively chosen subspace, $P_t(r)$ the best rank-$r$ projector of $g_t$, $\eta_{th}$ the information-retention threshold, $\hat g_t$ the projected gradient, $m_t,v_t$ the (subspace) first and second moments, $\beta_1,\beta_2$ their decay rates, and $\epsilon$ a stability constant.

Reference: Yehonathan Refael, Jonathan Svirsky, Boris Shustin, Wasim Huleihel, Ofir Lindenbaum, "AdaRankGrad: Adaptive Gradient-Rank and Moments for Memory-Efficient LLMs Training and Fine-Tuning", arXiv 2024. https://arxiv.org/abs/2410.17881

---
[Back to the Canon](../index.md)
