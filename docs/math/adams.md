# AdamS

Implements AdamS, an Adam variant that normalizes by the momentum itself instead of a separate second-moment estimate.

AdamS keeps Adam's exponential momentum but replaces the squared-gradient running average in the denominator with a blend of the squared previous momentum and the squared current gradient. This eliminates the second-moment state entirely, matching the memory footprint of SGD with momentum while retaining adaptive per-coordinate scaling. The denominator $\beta_2 m_{t-1}^2 + (1-\beta_2) g_t^2$ uses the previous momentum $m_{t-1}$ as a low-variance stand-in for the gradient scale. No bias correction is applied, and weight decay is decoupled.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 m_{t-1}^2 + (1-\beta_2) g_t^2 \\
\theta_t &= (1 - \eta \lambda)\, \theta_{t-1} - \eta \, \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the first moment, $v_t$ the momentum-based normalizer, $\beta_1,\beta_2$ the decay rates, $\lambda$ the weight decay, and $\epsilon$ a stability constant. All squares and the division act element-wise.

Reference: Huishuai Zhang, Bohan Wang, Luoxin Chen, "AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training", 2025. https://arxiv.org/abs/2505.16363

---
[Back to the Canon](../index.md)
