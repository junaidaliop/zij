# AdaLomo

Implements AdaLomo, low-memory optimization with adaptive learning rates.


$$
\begin{aligned}
\beta_{2,t} &= 1 - t^{c} \\
R_t &= \beta_{2,t} R_{t-1}
       + (1 - \beta_{2,t})\,(G_t^{\,2} + \epsilon_1 1_n 1_m^\top)\, 1_m \\
C_t &= \beta_{2,t} C_{t-1}
       + (1 - \beta_{2,t})\, 1_n^\top (G_t^{\,2} + \epsilon_1 1_n 1_m^\top) \\
\hat{V}_t &= R_t C_t / (1_n^\top R_t) \\
U_t &= G_t / \sqrt{\hat{V}_t}, \qquad
\hat{U}_t = U_t / \max\!\bigl(1, \mathrm{RMS}(U_t)/d\bigr) \\
\theta_t &= (1 - \eta_t \lambda)\,\theta_{t-1} - \eta_t\, \hat{U}_t,
\qquad \eta_t = \eta \max\!\bigl(\epsilon_2, \mathrm{RMS}(\theta_{t-1})\bigr)
\end{aligned}
$$

for a matrix parameter $\theta \in \mathbb{R}^{n \times m}$ with
gradient $G_t$, where $c$ is `decay_rate` (default
$-0.8$), $(\epsilon_1, \epsilon_2)$ is `eps`, $d$ is
`clip_threshold`, and $\lambda$ is the decoupled
`weight_decay`. Vector parameters keep an unfactored second moment.
As in LOMO, the update is computed and applied inside the backward pass,
so only the factored second moment of Adafactor (Shazeer and Stern,
ICML 2018) persists between steps.

Reference: Kai Lv, Hang Yan, Qipeng Guo, Haijun Lv, Xipeng Qiu,
"AdaLomo: Low-memory Optimization with Adaptive Learning Rate",
Findings of ACL 2024.
https://arxiv.org/abs/2310.10195


**Note:** Drive training with `fused_backward` instead of `loss.backward()` followed by `step()`. When `clip_grad_norm` is set, call `grad_norm` on the loss first. The second-moment buffers live outside `Optimizer.state` and are not captured by `state_dict`.


---
[Back to the Canon](../index.md)
