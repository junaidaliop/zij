# Variance-Adaptive Muon (Muon-NSR / Muon-VS)

Implements Variance-Adaptive Muon, a pair of Muon variants (Muon-NSR and Muon-VS) that rescale the momentum by its gradient-noise statistics before orthogonalization.

Muon orthogonalizes the momentum with Newton-Schulz iterations, but it ignores how noisy each coordinate of that momentum is. Variance-Adaptive Muon tracks a per-coordinate variance estimate $\Gamma_t$ of the gradient about the running mean and uses it to attenuate noisy directions. Crucially the rescaling is applied to the (Nesterov-extrapolated) momentum *before* the Newton-Schulz step, so the update keeps Muon's matrix-signed form rather than degenerating to a scalar adaptive method. The first variant, Muon-NSR, modulates by a noise-to-signal ratio with a tunable sensitivity $\gamma$; the second, Muon-VS, is the parameter-free limit that divides purely by the variance (recovered from Muon-NSR as $\gamma\,\hat{\Gamma}_t \gg \tilde{M}_t^{\odot 2}$).

$$
\begin{aligned}
M_t &= \beta\, M_{t-1} + (1-\beta)\, G_t \\
\Gamma_t &= \beta\, \Gamma_{t-1} + \beta(1-\beta)\,(M_{t-1} - G_t)^{\odot 2} \\
\hat{M}_t &= \frac{M_t}{1 - \beta^t}, \qquad \hat{\Gamma}_t = \frac{\Gamma_t}{1 - \beta^t} \\
\tilde{M}_t &= G_t + \frac{\beta}{1-\beta}\,\hat{M}_t \\
\bar{M}_{\mathrm{NSR},t} &= \frac{\tilde{M}_t}{\sqrt{\tilde{M}_t^{\odot 2} + \gamma\,\hat{\Gamma}_t + \epsilon}}, \qquad
\bar{M}_{\mathrm{VS},t} = \frac{\tilde{M}_t}{\sqrt{\hat{\Gamma}_t + \epsilon}} \\
O_t &= \mathrm{NS}_K(\bar{M}_t) \\
\theta_t &= \theta_{t-1}(1 - \eta\lambda) - \eta\, O_t
\end{aligned}
$$

where $\theta$ are the matrix-shaped parameters, $\eta$ the learning rate, $G_t$ the gradient, $M_t$ the momentum buffer with decay $\beta$, $\Gamma_t$ the variance tracker of the gradient about the prior mean, $\hat{M}_t,\hat{\Gamma}_t$ their bias-corrected forms, $\tilde{M}_t$ the Nesterov-extrapolated lookahead direction, $\gamma \ge 0$ the noise sensitivity (Muon-NSR), $\lambda$ the decoupled weight decay, $\epsilon$ a stability constant, and $\mathrm{NS}_K(\cdot)$ the polar factor from $K$ Newton-Schulz iterations (with $\bar{M}_t$ being $\bar{M}_{\mathrm{NSR},t}$ or $\bar{M}_{\mathrm{VS},t}$). All elementwise operations precede the orthogonalization so the step retains Muon's matrix-signed structure.

Reference: Jingru Li, Yibo Fan, Huan Li, "Variance-Adaptive Muon: Accelerating LLM Pretraining with NSR-Modulated and Variance-Scaled Momentum", arXiv 2026. https://arxiv.org/abs/2601.14603

---
[Back to the Canon](../README.md)
