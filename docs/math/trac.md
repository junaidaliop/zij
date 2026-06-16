# TRAC

Implements TRAC, a parameter-free scale tuner for any base optimizer.

TRAC keeps the reference point $\theta_{ref}$ (the parameters before
optimization began) and, after each base-optimizer update, rescales the
cumulative displacement by a learned scale $S_t$. To recover the base
optimizer's raw step direction, the displacement is first un-scaled by the
previous-step scale $S_{t-1}$, giving the un-scaled displacement
$\Delta_t = (\theta_t - \theta_{ref}) / (S_{t-1} + \epsilon)$. The scale
is the sum of $n$ one-dimensional discounted tuners, one per discount
factor $\beta_i$. With base update producing $\theta_t$, gradient
$g_t$, and inner product $h_t$:


$$
\begin{aligned}
\Delta_t &= \frac{\theta_t - \theta_{ref}}{S_{t-1} + \epsilon} \\
h_t &= \langle g_t,\, \Delta_t \rangle \\
v_{t,i} &= \beta_i^{2}\, v_{t-1,i} + h_t^{2} \\
\sigma_{t,i} &= \beta_i\, \sigma_{t-1,i} - h_t \\
s_{t,i} &= \frac{s_{init}}{\mathrm{erfi}(1/\sqrt{2})}\,
    \mathrm{erfi}\!\Bigl(\frac{\sigma_{t,i}}{\sqrt{2 v_{t,i}} + \epsilon}\Bigr) \\
S_t &= \max\Bigl(0,\, \sum_{i=1}^{n} s_{t,i}\Bigr) \\
\theta_{t+1} &= \theta_{ref} + S_t\,\Delta_t
\end{aligned}
$$

where $\mathrm{erfi}$ is the imaginary error function and
$s_{init}$ is the initial scale `s_prev`.

Reference: Aneesh Muppidi, Zhiyu Zhang, Heng Yang,
"Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning",
NeurIPS 2024.
https://arxiv.org/abs/2405.16642


**Note:** this is a wrapper around a base optimizer. Pass an already constructed

optimizer instance, e.g.
`TRAC(torch.optim.AdamW(model.parameters()))`.

---
[Back to the Canon](../index.md)
