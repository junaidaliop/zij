# ZetA

Implements ZetA, a hybrid optimizer that blends an Adam update with a Riemann-zeta-scaled step.

ZetA forms the usual Adam direction $u_{\text{adam}}$ from bias-corrected moments, then adds a second direction $u_\zeta$ whose magnitude is controlled by the Riemann zeta function $\zeta(s_t)$ at a time-varying exponent $s_t \in (1, 2]$. The zeta term divides the bias-corrected first moment by a power of the gradient norm and by $\zeta(s_t)$, and is amplified by a boost factor $b_t$ that grows when consecutive gradients are positively aligned. The two directions are mixed by $\alpha$, the step is taken under a cosine learning-rate schedule with decoupled weight decay, and a SAM-style perturbation precedes the update.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{t}} \\
s_t &= s_{\min} + (s_{\max}-s_{\min})\Big(1 - \big|1 - \tfrac{2(t \bmod T)}{T}\big|\Big) \\
b_t &= 1 + \delta_t \cdot 0.2 \cdot \max\!\Big(0,\ \frac{\langle g_t, g_{t-1}\rangle}{\lVert g_t\rVert\,\lVert g_{t-1}\rVert + \epsilon}\Big) \\
u_{\text{adam}} &= \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} \\
u_\zeta &= \eta\,\hat m_t\, b_t \cdot \frac{1}{\lVert g_t\rVert^{\,s_t-1} + \epsilon} \cdot \frac{1}{\zeta(s_t)} \\
u_t &= \alpha\, u_{\text{adam}} + (1-\alpha)\, u_\zeta \\
\eta_t &= \eta \cdot \tfrac{1}{2}\big(1 + \cos\tfrac{\pi t}{T}\big)\,(1 - \lambda\,\eta_t) \\
\theta_t &= \theta_{t-1} - \eta_t\, u_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first- and second-moment estimates, $\hat m_t$/$\hat v_t$ their bias-corrected forms, $\beta_1,\beta_2$ the decay rates, $\lambda$ the weight decay, $\epsilon$ a small stability constant, $\zeta(\cdot)$ the Riemann zeta function, $s_t \in (s_{\min}, s_{\max}] \subset (1,2]$ the dynamic zeta exponent, $b_t$ the cosine-similarity boost ($\delta_t$ a boost gate), $\alpha$ the mix between the Adam and zeta directions, and $T$ the total number of steps. A SAM-style perturbation $\theta^{+} = \theta + \gamma\, u_t / (\lVert u_t\rVert + \epsilon)$ is applied before computing the final update.

Reference: Samiksha BC, "ZetA: A Hybrid Optimizer Combining Riemann Zeta Scaling with Adam for Robust Deep Learning", arXiv 2025. https://arxiv.org/abs/2508.02719

---
[Back to the Canon](../README.md)
