# Ringleader ASGD

Implements Ringleader ASGD, an asynchronous SGD where a central ringleader aggregates stale worker gradients to achieve optimal time complexity under data heterogeneity.

Workers compute stochastic gradients at possibly outdated parameter snapshots and stream them to a server, which accumulates them in a table. Each round, the server forms the estimator by first averaging all gradients received from each worker, then averaging across workers, and takes a single descent step. Structured buffering keeps every delay bounded by $2n-2$, so no computed gradient is ever wasted.

$$
\begin{aligned}
g_i^{k,j} &:= \nabla f_i\bigl(\theta^{k-\delta_i^k}; \xi_i^{k-\delta_i^k,j}\bigr), \\
\bar{g}^k &:= \frac{1}{n}\sum_{i=1}^{n} \frac{1}{b_i^k}\sum_{j=1}^{b_i^k} g_i^{k,j}, \\
\theta^{k+1} &= \theta^k - \gamma\, \bar{g}^k.
\end{aligned}
$$

where $\theta$ are the model parameters, $\gamma$ is the stepsize, $n$ is the number of workers, $b_i^k$ is the number of gradients accumulated from worker $i$ at iteration $k$, $\delta_i^k \le 2n-2$ is the bounded delay of worker $i$, and $\xi$ denotes the data sample. With smoothness $L$, variance $\sigma^2$, target accuracy $\epsilon$, and harmonic mean batch size $B^k = \bigl(\tfrac{1}{n}\sum_{i=1}^n 1/b_i^k\bigr)^{-1}$, the stepsize is $\gamma = \min\{1/(8nL),\ \epsilon B/(10L\sigma^2)\}$.

Reference: Artavazd Maranjyan, Peter Richtárik, "Ringleader ASGD: The First Asynchronous SGD with Optimal Time Complexity under Data Heterogeneity", arXiv 2025. https://arxiv.org/abs/2509.22860

---
[Back to the Canon](../index.md)
