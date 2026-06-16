# AdamWScheduleFree

Implements Schedule-Free AdamW, AdamW with momentum replaced by interpolation and averaging.


$$
\begin{aligned}
    y_t &= (1 - \beta_1) z_t + \beta_1 x_t, \\
    v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \qquad
        \hat{v}_t = v_t / (1 - \beta_2^t), \\
    z_{t+1} &= z_t - \gamma_t \left( \frac{g_t}{\sqrt{\hat{v}_t}
        + \epsilon} + \lambda y_t \right), \\
    x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},
\end{aligned}
$$

where gradients $g_t$ are evaluated at the interpolated point
$y_t$, the parameters used for evaluation are the average
$x_t$, and $c_{t+1} = \gamma_t^2 / \sum_{i=1}^{t}
\gamma_i^2$. No learning rate schedule is needed; linear warmup is
available through `warmup_steps`.

Reference: Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko,
Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
https://arxiv.org/abs/2405.15682


**Note:** Call `optimizer.train()` before training and `optimizer.eval()` before evaluation or checkpointing, alongside the matching `model.train()` / `model.eval()` calls. Gradients are computed at $y_t$ while losses should be measured at $x_t$, so the parameter buffer must be switched between the two points.


---
[Back to the Canon](../index.md)
