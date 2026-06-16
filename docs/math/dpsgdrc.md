# DP-SGD-RC

Implements DP-SGD-RC, differentially private SGD with randomized per-sample clipping.

Standard DP-SGD clips every per-sample gradient to a fixed norm $C$ and adds Gaussian noise, but computing exact per-sample norms is the memory bottleneck when fine-tuning large language models on long contexts. DP-SGD-RC ("Randomized Clipping") replaces the exact per-sample norm with a cheap stochastic estimate $\hat{n}_i$ obtained from trace estimation (Hutchinson / Hutch++), so the clipping factor becomes random while the rest of the DP-SGD pipeline—noise calibrated to the clip threshold $C$ and a standard optimizer step—is unchanged. Because the squared norm is estimated, the clip factor uses $C/\sqrt{\hat{n}_i}$.

$$
\begin{aligned}
\bar{g}_i &= \min\!\left(1, \frac{C}{\sqrt{\hat{n}_i}}\right) g_i, \\
\tilde{g} &= \frac{1}{L}\left(\sum_{i=1}^{L} \bar{g}_i + \sigma C \, \mathcal{N}(0, I)\right), \\
\theta_{t+1} &= \theta_t - \eta \, \tilde{g}.
\end{aligned}
$$

where $g_i$ is the per-sample gradient, $\hat{n}_i$ is the randomized estimate of $\|g_i\|^2$, $C$ is the clipping threshold, $L$ is the lot size, $\sigma$ is the noise multiplier, $\mathcal{N}(0,I)$ is Gaussian noise, $\eta$ is the learning rate, and $\theta$ the parameters (the descent step may be replaced by Adam).

Reference: Enayat Ullah, Sai Aparna Aketi, Devansh Gupta, Huanyu Zhang, Meisam Razaviyayn, "Efficient DP-SGD for LLMs with Randomized Clipping", arXiv 2025. https://arxiv.org/abs/2605.24879

---
[Back to the Canon](../index.md)
