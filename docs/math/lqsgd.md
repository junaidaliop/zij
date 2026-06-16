# LQ-SGD

Implements LQ-SGD, communication-efficient distributed SGD that combines low-rank gradient factorization with logarithmic quantization and error feedback.

LQ-SGD targets the communication bottleneck in distributed training. Each step adds the previous round's compression residual back to the gradient (error feedback), factorizes the corrected gradient into low-rank factors $P_t$ and $Q_t$ via one power-iteration step, and transmits those factors after a logarithmic quantization that allocates more resolution to the small magnitudes dominating the gradient distribution. The reconstructed gradient drives a plain SGD step, and the reconstruction residual is carried forward.

$$
\begin{aligned}
G_t' &= G_t + E_{t-1} \\
P_t &= \mathrm{Orthonormalize}(G_t' Q_t) \\
P_t &= \mathrm{LogDeq}(\mathrm{LogQuant}(P_t, b_p, \alpha)) \\
Q_t &= G_t'^{\top} P_t, \quad Q_t = \mathrm{LogDeq}(\mathrm{LogQuant}(Q_t, b_q, \alpha)) \\
\hat{G}_t &= P_t Q_t^{\top} \\
E_t &= G_t' - \hat{G}_t \\
\theta_{t+1} &= \theta_t - \eta\, \hat{G}_t
\end{aligned}
$$

where $G_t$ is the local gradient matrix, $E_t$ the error-feedback buffer, $\eta$ the learning rate, and $\alpha > 0$ controls the curvature of the logarithmic codec $q(x) = \mathrm{sign}(x)\,\dfrac{\log(1+\alpha|x|)}{\log(1+\alpha)}$ with inverse $\mathrm{sign}(q)\,\dfrac{(1+\alpha)^{|q|}-1}{\alpha}$; $b_p, b_q$ are the bit-widths for the two factors. The quantized factors are exchanged across workers via All-Reduce; no momentum is used.

Reference: Hongyang Li, Lincen Bai, Caesar Wu, Mohammed Chadli, Said Mammar, Pascal Bouvry, "Trustworthy Efficient Communication for Distributed Learning using LQ-SGD Algorithm", arXiv 2025. https://arxiv.org/abs/2506.17974

---
[Back to the Canon](../index.md)
