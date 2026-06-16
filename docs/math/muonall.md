# MuonAll

Implements MuonAll, a Muon variant that routes every parameter, including 1D ones, through the orthogonalized update.

Standard Muon applies its Newton-Schulz orthogonalization only to 2D weight matrices and leaves 1D parameters (biases, norms, embeddings) to a separate optimizer. MuonAll instead embeds each 1D parameter as the main diagonal of a square matrix, runs it through the same momentum-plus-orthogonalization pipeline, and maps the result back to a vector, so a single rule governs all parameters during finetuning.

$$
\begin{aligned}
B_t &\leftarrow \mu B_{t-1} + g_t \\
O_t &\leftarrow \mathrm{NewtonSchulz5}(g_t + \mu B_t) \\
\theta_t &\leftarrow \theta_{t-1} - \eta\, O_t
\end{aligned}
$$

where $\theta$ are the parameters (a 1D parameter is first reshaped to $\mathrm{diag}(\theta)$ and the orthogonalized update reshaped back), $g_t$ is the gradient, $B_t$ is the momentum buffer with decay $\mu$, $\eta$ is the learning rate, and $\mathrm{NewtonSchulz5}$ is the quintic Newton-Schulz iteration that approximates the orthogonal factor $(M_t M_t^\top)^{-1/2} M_t = U V^\top$.

Reference: Saurabh Page, Advait Joshi, S. S. Sonawane, "MuonAll: Muon Variant for Efficient Finetuning of Large Language Models", arXiv 2025. https://arxiv.org/abs/2511.06086

---
[Back to the Canon](../index.md)
