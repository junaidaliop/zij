# HybridSGD

Implements HybridSGD, a 2D-parallel SGD that nests s-step SGD inside federated averaging across a processor grid.

The processor mesh is partitioned into $p_r$ row teams of $p_c$ processors each. Within a row team, communication is deferred by computing $s$ gradient steps as a single batched update (s-step SGD). Across row teams, local models are synchronized every $\tau$ iterations by averaging (FedAvg). The underlying per-step rule is plain SGD; the two dimensions trade communication frequency ($\tau$) against the deferral depth ($s$).

$$
\begin{aligned}
x_{sk+s} &= x_{sk} - \eta \cdot Y^{\top} \begin{bmatrix} u_{sk+1} \\ \vdots \\ u_{sk+s} \end{bmatrix} \\
x_{k} &= \frac{1}{p}\sum_{i=1}^{p} \tilde{x}_{k}^{[i]}
\end{aligned}
$$

where $\eta$ is the learning rate, $x$ the model parameters, $Y$ the stacked $s$-step sampling matrix combining the deferred gradient contributions, $u_{sk+j}$ the corrected gradient terms for the postponed updates, $\tilde{x}_{k}^{[i]}$ the locally updated model on processor $i$ after $\tau$ local iterations, and $p$ the number of processors averaged.

Reference: Aditya Devarakonda, Ramakrishnan Kannan, "Communication-Efficient, 2D Parallel Stochastic Gradient Descent for Distributed-Memory Optimization", arXiv preprint 2025. https://arxiv.org/abs/2501.07526

---
[Back to the Canon](../index.md)
