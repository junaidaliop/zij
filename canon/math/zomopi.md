# ZO-MOPI

Implements ZO-MOPI, a zeroth-order spectral optimizer that orthogonalizes a low-rank momentum via streaming power iteration.

ZO-MOPI is a derivative-free analogue of Muon for matrix-shaped parameters. Gradients are estimated from forward passes alone, using a low-rank subspace perturbation $\mathbf{A}\mathbf{B}_t^{(i)}$: the loss is probed at $\mathbf{X}_t \pm \mu\mathbf{A}\mathbf{B}_t^{(i)}$ and the central finite-difference quotient reweights $\mathbf{B}_t^{(i)}$, giving an estimator that lives in the reduced $\mathbb{R}^{r\times n}$ space. The shared sketch $\mathbf{A}\in\mathbb{R}^{m\times r}$ is resampled every $\nu$ iterations (and momentum reprojected), which keeps both the query cost and the memory low.

Instead of Muon's Newton-Schulz iteration, which equalizes the entire spectrum and can amplify noisy directions, ZO-MOPI applies partial orthogonalization: a warm-started streaming power iteration extracts only the top-$k$ singular directions of the momentum and replaces them with an orthonormal frame, producing a rank-$k$ approximate matrix sign $\mathbf{U}_t\mathbf{V}_t^{\top}$. Caching the right singular vectors $\mathbf{V}_{t-1}$ across steps lets a single power-iteration sweep suffice per update.

$$
\begin{aligned}
\hat{\mathbf{G}}_t &= \frac{1}{N}\sum_{i=1}^{N} \frac{F(\mathbf{X}_t + \mu\mathbf{A}\mathbf{B}_t^{(i)};\,\xi) - F(\mathbf{X}_t - \mu\mathbf{A}\mathbf{B}_t^{(i)};\,\xi)}{2\mu}\, \mathbf{B}_t^{(i)}, \qquad \mathbf{B}_t^{(i)} \sim \mathcal{N}(0, I) \\
\mathbf{M}_t &= \beta\,\mathbf{M}_{t-1} + (1-\beta)\,\hat{\mathbf{G}}_t \\
\mathbf{V}_t &= \mathrm{QR}\!\left(\mathbf{M}_t^{\top}(\mathbf{M}_t \mathbf{V}_{t-1})\right), \qquad \mathbf{U}_t = \mathrm{NormalizeColumns}(\mathbf{M}_t \mathbf{V}_t) \\
\mathbf{O}_t &= \mathbf{U}_t \mathbf{V}_t^{\top} \\
\mathbf{X}_t &= \mathbf{X}_{t-1} - \eta\,\mathbf{A}\,\mathbf{O}_t
\end{aligned}
$$

where $\mathbf{X}_t\in\mathbb{R}^{m\times n}$ is the parameter matrix, $F(\cdot;\xi)$ is the stochastic loss, $\mathbf{A}\in\mathbb{R}^{m\times r}$ is the random subspace sketch (resampled every $\nu$ steps), $\mathbf{B}_t^{(i)}\in\mathbb{R}^{r\times n}$ are the Gaussian probe directions, $\mu$ is the smoothing radius, $N$ is the number of queries, $\beta$ is the momentum coefficient, $\eta$ is the learning rate, $\mathbf{M}_t\in\mathbb{R}^{r\times n}$ is the low-rank momentum, $k$ is the spectral rank of the partial orthogonalization, and $\mathbf{O}_t = \mathbf{U}_t\mathbf{V}_t^{\top}$ is the rank-$k$ approximate matrix sign returned by the streaming power iteration over cached singular vectors $\mathbf{V}_{t-1}$.

Reference: Jiahe Chen, Ziye Ma, "Accelerating Zeroth-Order Spectral Optimization with Partial Orthogonalization from Power Iteration", arXiv preprint 2026. https://arxiv.org/abs/2605.09034

---
[Back to the Canon](../README.md)
