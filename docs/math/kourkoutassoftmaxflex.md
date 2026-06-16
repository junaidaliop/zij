# KourkoutasSoftmaxFlex

Implements Kourkoutas-beta, an Adam variant with a layer-wise dynamic
$\beta_2$ driven by a bounded "sunspike" ratio.

For each layer the optimizer tracks an exponential moving average of the
pooled gradient norm and compares the current norm against it. A large
ratio (a gradient spike) lowers $\beta_2$ toward $\beta_{2,\min}$
so the second moment reacts faster; a calm phase keeps $\beta_2$ near
$\beta_{2,\max}$, recovering Adam-like behavior.


$$
\begin{aligned}
     n_t &= \lVert g_t \rVert_2                                            \\
     e_t &= \alpha\, e_{t-1} + (1 - \alpha)\, n_t                          \\
     r_t &= \frac{n_t}{e_t + \tau}                                         \\
     s_t &= \frac{r_t}{1 + r_t}                                            \\
     \beta_{2,t} &= \beta_{2,\max}
         - (\beta_{2,\max} - \beta_{2,\min})\, s_t                         \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
     v_t &= \beta_{2,t} v_{t-1} + (1 - \beta_{2,t}) g_t^2                  \\
     \theta_t &= \theta_{t-1} - \eta\,
         \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

The norm $n_t$ is pooled over every parameter in a layer, so the
sunspike $s_t \in [0, 1)$ and the resulting $\beta_{2,t}$ are
shared by all tensors of that layer. During the first `warmup_steps` the
sunspike is held at zero and $\beta_2$ is fixed at the midpoint
$\tfrac{1}{2}(\beta_{2,\min} + \beta_{2,\max})$. The constant
$\tau$ is `tiny_spike`.

Optional features (all off by default) are leaky-AMSGrad on the second
moment (`decay`), a trust-region clip $\lvert \Delta\theta \rvert
\le \eta \cdot \mathrm{max\_ratio}$ (`max_ratio`), an adaptive tiny term
that scales the denominator floor with $\langle \lvert\theta\rvert
\rangle$ (`adaptive_tiny`), and bias correction (`bias_correction`).
With all features off, `bias_correction="none"`, and
$\beta_{2,\min} = \beta_{2,\max}$, the method reduces to Adam.


**Note:** Each parameter group is treated as one layer: the sunspike ratio and $\beta_2$ are pooled across the group's parameters. Split the parameters into separate groups to obtain finer-grained layer-wise $\beta_2$.

Reference: Stavros C. Kassinos, "Kourkoutas-Beta: A Sunspike-Driven Adam
Optimizer with Desert Flair", 2025.
https://arxiv.org/abs/2508.12996

---
[Back to the Canon](../index.md)
