# Ranger21

Implements Ranger21, a synergistic combination of AdamW and eight techniques.

Ranger21 keeps an AdamW core and layers on adaptive gradient clipping,
gradient centralization, gradient normalization, positive-negative
momentum, norm loss, stable weight decay, a linear warmup combined with an
explore-exploit warmdown schedule, Lookahead, and a softplus-smoothed
denominator. The positive-negative momentum keeps two first-moment buffers,
one for odd and one for even steps, and forms the update direction as a
positively weighted current moment minus a negatively weighted previous
moment, normalized so the learning rate need not change with `beta0`:


$$
\begin{aligned}
     m_t &= \beta_1^2 m_{t-2} + (1 - \beta_1^2) g_t                       \\
     \hat{m}_t &= \frac{(1 + \beta_0) m_t - \beta_0 m_{t-1}}
         {\sqrt{(1 + \beta_2)^2 + \beta_2^2}}                              \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,\quad
         v_t \leftarrow \max(v_t^{\max}, v_t)                             \\
     \theta_t &= \theta_{t-1} - \frac{\eta_t}{1 - \beta_1^t} \,
         \frac{\hat{m}_t}{\sqrt{v_t} / \sqrt{1 - \beta_2^t} + \epsilon}
\end{aligned}
$$

The learning rate $\eta_t$ follows the explore-exploit schedule, a
linear warmup over the first $t_{\text{warmup}}$ steps, a flat phase,
and a linear warmdown over the last $t_{\text{warmdown}}$ steps, which
is why `num_iterations` (the total number of training steps) is required.


**Note:** Following the reference implementation, the positive-negative momentum combination fixes the coefficients to $\beta_0 = 1$ (so the update is $2 m_t - m_{t-1}$) and normalizes by $\sqrt{(1 + \beta_2)^2 + \beta_2^2}$; the `beta0` argument is retained only for the noise-amplitude validation range.

Reference: Less Wright, Nestor Demeure,
"Ranger21: a synergistic deep learning optimizer", arXiv 2021.
https://arxiv.org/abs/2106.13731

---
[Back to the Canon](../index.md)
