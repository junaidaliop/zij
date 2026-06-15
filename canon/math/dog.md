# DoG

Implements DoG, the parameter-free distance-over-gradients step size schedule.


$$
\begin{aligned}
\bar{r}_t &= \max\bigl(\bar{r}_{t-1},\, \lVert \theta_t - \theta_0 \rVert\bigr) \\
G_t &= \sum_{i \le t} \lVert g_i \rVert^2 \\
\eta_t &= \frac{\bar{r}_t}{\sqrt{G_t}} \\
\theta_{t+1} &= \theta_t - \eta_t\, g_t
\end{aligned}
$$

where the initial distance estimate is
$\bar{r}_0 = r_\epsilon = \alpha\,(1 + \lVert \theta_0 \rVert)$ with
$\alpha$ given by `reps_rel`, and `lr` enters only as a constant
multiplier $c$ on $\eta_t$.


**Note:** Leave `lr` at its default of 1.0. The paper recommends pairing DoG with polynomial decay iterate averaging, and raising `reps_rel` to 1e-4 for models that use batch normalization.

Reference: Maor Ivgi, Oliver Hinder, Yair Carmon,
"DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule",
ICML 2023.
https://arxiv.org/abs/2302.12022

---
[Back to the Canon](../README.md)
