# Cautious Weight Decay

Implements Cautious Weight Decay, a sign-aware modification that applies decoupled weight decay only where it does not fight the optimizer's own update.

Standard decoupled weight decay shrinks every coordinate toward zero each step, which can pull a parameter against the direction the base optimizer is driving it. Cautious Weight Decay (CWD) gates the decay term with an indicator mask that is active only on coordinates where the update direction $u_t$ and the parameter $\theta_t$ share the same sign (their elementwise product is non-negative). On those coordinates decay and the update agree, so shrinking is harmless; elsewhere decay is suppressed. The mask wraps any base update direction $u_t$, so it drops into SGD, AdamW, and similar methods unchanged. For AdamW, $u_t = \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta\bigl(u_t + \lambda\,\mathbb{I}(u_t \odot \theta_t \ge 0)\odot\theta_t\bigr), \\
u_t^{\mathrm{AdamW}} &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta$ the learning rate, $\lambda$ the weight-decay coefficient, $u_t$ the base optimizer's update direction, $\odot$ elementwise multiplication, and $\mathbb{I}(\cdot)$ the elementwise indicator that is $1$ where $u_t \odot \theta_t \ge 0$ and $0$ otherwise; $\hat{m}_t$, $\hat{v}_t$ are the bias-corrected first and second moments.

Reference: Lizhang Chen, Jonathan Li, Kaizhao Liang, Baiyu Su, Cong Xie, Nuo Wang Pierse, Chen Liang, Ni Lao, Qiang Liu, "Cautious Weight Decay", arXiv 2025. https://arxiv.org/abs/2510.12402

---
[Back to the Canon](../README.md)
