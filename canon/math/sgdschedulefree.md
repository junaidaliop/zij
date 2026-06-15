# SGDScheduleFree

Implements Schedule-Free SGD, which replaces momentum with interpolation and averaging.


$$
\begin{aligned}
    y_t &= (1 - \beta) z_t + \beta x_t, \\
    z_{t+1} &= z_t - \gamma_t \bigl(\nabla f(y_t) + \lambda y_t\bigr), \\
    x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1},
\end{aligned}
$$

where $z_t$ is the base SGD iterate, gradients are evaluated at the
interpolated point $y_t$, the parameters used for evaluation are the
average $x_t$, $\lambda$ is `weight_decay`, and
$c_{t+1} = \gamma_t^2 / \sum_{i=1}^{t} \gamma_i^2$. No learning
rate schedule is needed; linear warmup is available through
`warmup_steps`.

Reference: Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko,
Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
https://arxiv.org/abs/2405.15682


**Note:** Call `optimizer.train()` before training and `optimizer.eval()` before evaluation or checkpointing, alongside the matching `model.train()` / `model.eval()` calls. Gradients are computed at $y_t$ while losses should be measured at $x_t$, so the parameter buffer must be switched between the two points.


---
[Back to the Canon](../README.md)
