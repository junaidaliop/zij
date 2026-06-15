# Amos

Implements Amos, an Adam-style optimizer with adaptive weight decay
towards a model-oriented scale.

Amos replaces the tuned weight decay of Adam with a decay schedule driven by
a per-variable model-oriented scale $\xi$, an estimate of the magnitude
each weight should settle at. The second moment is a scalar mean of the
squared gradient, so the running buffers are size one per parameter tensor.


$$
\begin{aligned}
     \tilde{v}_t &= \beta\, \tilde{v}_{t-1}
         + (1 - \beta)\, \overline{g_t^2}                                 \\
     r_{v,t} &= \frac{1 - \beta^t}{\tilde{v}_t + \epsilon}                \\
     c_t &= \frac{1}{\sqrt{1 + c\,\sqrt{\eta}\; b_{t-1}}}                  \\
     d_t &= \frac{1}{1 + d\,\sqrt{\eta\,\xi}\; b_{t-1}}                    \\
     \gamma_t &= c_t\, \eta^2\, r_{v,t}\, \overline{g_t^2}                \\
     \theta_t &= \theta_{t-1} - d_t \left(
         \eta\,\xi\,\sqrt{r_{v,t}}\; g_t
         + \bigl(\tfrac{1}{2}\gamma_t + \lambda\bigr)\, \theta_{t-1} \right)  \\
     b_t &= b_{t-1}\,(1 + \gamma_t) + \gamma_t
\end{aligned}
$$

where $\overline{g_t^2}$ is the mean of the squared gradient over the
parameter tensor, $\xi$ is the model-oriented scale returned by
`get_scale`, $b_t$ is the accumulated decay buffer, $c$ and
$d$ are the decay coefficients `c_coef` and `d_coef`, and
$\lambda$ is the additional L2 term `extra_l2`. An optional moving
average of the update with rate `momentum` is applied before the step.

Reference: Ran Tian, Ankur P. Parikh, "Amos: An Adam-style Optimizer with
Adaptive Weight Decay towards Model-Oriented Scale", 2022.
https://arxiv.org/abs/2210.11693

---
[Back to the Canon](../README.md)
