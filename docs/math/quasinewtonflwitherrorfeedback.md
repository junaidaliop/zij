# Quasi-Newton FL with Error Feedback

Implements EF21+L-BFGS, a quasi-Newton federated learning method that augments EF21 error feedback with an L-BFGS search direction.

EF21 controls communication by having each client transmit only a compressed gradient difference $\mathcal{C}(\nabla f_i(x^t) - g_i^{t-1})$, updating a local gradient state $g_i^t$ whose average $g^t$ tracks the true gradient despite biased compression. The baseline EF21 server then takes a fixed-stepsize step $x^{t+1} = x^t - \gamma g^t$. This method replaces that first-order step with a quasi-Newton one: the server keeps a memory of $m$ pairs $(s^k, y^k)$ formed from successive iterates and aggregated states, and runs the standard L-BFGS two-loop recursion on the aggregated state $g^t$ to produce a search direction $p^t$ that captures curvature, then updates the model with a line-searched stepsize.

For client $i$ holding $f_i$ and a server coordinating $M$ clients, with $s^k = x^{k+1}-x^k$ and $y^k = g^{k+1}-g^k$, the update is:

$$
\begin{aligned}
g_i^t &= g_i^{t-1} + \mathcal{C}\!\left(\nabla f_i(x^t) - g_i^{t-1}\right), \\
g^t &= \frac{1}{M}\sum_{i=1}^{M} g_i^t, \\
p^t &= \mathrm{TwoLoop}\!\left(g^t,\ \{(s^k, y^k)\}_{k=t-m}^{t-1}\right), \\
x^{t+1} &= x^t - \alpha^t\, p^t,
\end{aligned}
$$

where $\mathrm{TwoLoop}$ is the L-BFGS two-loop recursion that applies the implicit inverse-Hessian approximation to $g^t$: initializing $p \leftarrow g^t$, looping $k=t-1,\dots,t-m$ with $\alpha_k = \tfrac{s^k\cdot p}{s^k\cdot y^k}$ and $p \leftarrow p - \alpha_k y^k$, scaling $p \leftarrow \tfrac{s^{t-1}\cdot y^{t-1}}{y^{t-1}\cdot y^{t-1}}\,p$, then looping $k=t-m,\dots,t-1$ with $\beta = \tfrac{y^k\cdot p}{s^k\cdot y^k}$ and $p \leftarrow p + (\alpha_k-\beta)s^k$.

where $x^t$ are the global model parameters, $\nabla f_i(x^t)$ the local gradient at client $i$, $\mathcal{C}$ a biased (contractive) compressor satisfying $\mathbb{E}\,\|\mathcal{C}(x)-x\|^2 \le (1-\alpha)\|x\|^2$ (for example Top-$K$), $g_i^t$ the per-client gradient state and $g^t$ its server-side average, $m$ the memory size, $(s^k, y^k)$ the iterate/gradient-state difference pairs, $p^t$ the L-BFGS search direction, $\gamma$ the EF21 baseline stepsize, and $\alpha^t$ the (line-searched) stepsize.

Reference: Yanlin Wu, Dmitry Kamzolov, Martin Takáč, "Quasi-Newton Methods for Federated Learning with Error Feedback", OPT2025: 17th Annual Workshop on Optimization for Machine Learning (NeurIPS workshop) 2025. https://opt-ml.org/papers/2025/paper148.pdf

---
[Back to the Canon](../index.md)
