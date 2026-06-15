# Power Decay / Warmup-Stable-Decay (WSD)

Implements Power Decay and Warmup-Stable-Decay (WSD), optimal learning-rate schedules derived from functional scaling laws.

These schedules are the provably optimal learning-rate trajectories under a functional scaling-law analysis of the training loss, where the optimal form is governed by two task exponents: a source exponent $s$ (smaller means a harder task) and a capacity exponent $\beta$ (smaller means higher model capacity). In the easy-task regime ($s \ge 1 - 1/\beta$) the optimal schedule is a single power decay from a peak rate to zero. In the hard-task regime ($s < 1 - 1/\beta$) the optimal schedule is Warmup-Stable-Decay: hold the rate at the maximum stable value, then power-decay over a vanishing terminal fraction of training. Both share the same decay exponent $2\beta - 1$.

$$
\begin{aligned}
\eta_t^{\text{power}} &= \eta_{\text{peak}}\left(1 - \frac{t}{T}\right)^{2\beta - 1}, \\
\eta_t^{\text{wsd}} &=
\begin{cases}
\eta_{\text{stab}}, & 0 \le t \le T_1, \\
\eta_{\text{stab}}\left(1 - \dfrac{t - T_1}{T - T_1}\right)^{2\beta - 1}, & T_1 < t \le T,
\end{cases}
\end{aligned}
$$

where $t$ is the training step, $T$ the total training horizon, $T_1$ the breakpoint where the decay phase begins, $\eta_{\text{peak}}$ the peak learning rate, $\eta_{\text{stab}}$ the maximum stable learning rate, and $\beta > 1$ the capacity exponent setting the decay power $2\beta - 1$. The decay fraction $(T - T_1)/T \to 0$ as $T$ grows.

Reference: Binghui Li, Zilin Wang, Fengling Chen, Shiyang Zhao, Ruiheng Zheng, Lei Wu, "Optimal Learning-Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay", arXiv 2025. https://arxiv.org/abs/2602.06797

---
[Back to the Canon](../README.md)
