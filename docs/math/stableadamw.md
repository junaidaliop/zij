# StableAdamW

Implements StableAdamW, AdamW with Adafactor-style update clipping.

StableAdamW rescales each AdamW update by the root-mean-square of the
per-coordinate ratio between the gradient and the second-moment estimate.
When that ratio is large the effective learning rate is shrunk, which
removes the loss spikes that gradient clipping leaves behind during
low-precision training.


$$
\begin{aligned}
     \mathrm{debias\_beta}(\beta, t) &=
         \frac{\beta^t - \beta}{\beta^t - 1}                              \\
     c_1(t) &= 1 - \mathrm{debias\_beta}(\beta_1, t)                      \\
     c_2(t) &= \mathrm{debias\_beta}(\beta_2, t)                          \\
     m_t &= (1 - c_1(t)) \, m_{t-1} + c_1(t) \, g_t                       \\
     v_t &= c_2(t) \, v_{t-1} + (1 - c_2(t)) \, g_t^2                      \\
     \mathrm{RMS}_t &= \sqrt{\,
         \mathrm{mean}\!\left(
             \frac{g_t^2}{\max(v_t, \epsilon^2)}
         \right)}                                                         \\
     \eta_t &= \frac{\eta}{\max(1, \mathrm{RMS}_t)}                       \\
     \theta_t &= \theta_{t-1} - \eta_t \,
         \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

Bias correction is applied through the step-dependent coefficients
$c_1(t)$ and $c_2(t)$, which are computed from
$\mathrm{debias\_beta}(\beta, t) = (\beta^t - \beta) / (\beta^t - 1)$
rather than via a separate $(1 - \beta^t)$ normalization. The moments
are therefore updated and consumed in their interpolated (lerp) form.
Weight decay is decoupled by default. Optional Kahan summation compensates
for rounding when the parameters are stored in `float16` or `bfloat16`.

Reference: Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos,
Ali Farhadi, Ludwig Schmidt, "Stable and low-precision training for
large-scale vision-language models", NeurIPS 2023.
https://arxiv.org/abs/2304.13013

---
[Back to the Canon](../index.md)
