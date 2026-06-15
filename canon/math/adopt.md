# ADOPT

Implements ADOPT, a modified Adam that converges with any $\beta_2$.

ADOPT removes the current gradient from the second-moment estimate used to
normalize the update and swaps the order of the momentum update and the
normalization, which gives an optimal convergence rate without tuning
$\beta_2$. With $v_0 = g_0^2$ and $m_0 = 0$, each step
$t \geq 1$ computes:


$$
\begin{aligned}
     \hat{g}_t &= \mathrm{clip}\!\left(
         \frac{g_t}{\max(\sqrt{v_{t-1}}, \epsilon)},\;
         -c_t,\; c_t \right) \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \hat{g}_t \\
     \theta_t &= \theta_{t-1} - \eta\, m_t \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2
\end{aligned}
$$

where the clipping bound defaults to $c_t = t^{1/4}$, set through
`clip_lambda`. Passing `clip_lambda=None` recovers the unclipped ADOPT.

Reference: Shohei Taniguchi, Keno Harada, Gouki Minegishi, Yuta Oshima,
Seong Cheol Jeong, Go Nagahara, Tomoshi Iiyama, Masahiro Suzuki,
Yusuke Iwasawa, Yutaka Matsuo,
"ADOPT: Modified Adam Can Converge with Any $\beta_2$ with the Optimal
Rate", NeurIPS 2024.
https://arxiv.org/abs/2411.02853

---
[Back to the Canon](../README.md)
