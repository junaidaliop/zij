# FedRepOpt

Implements FedRepOpt, a gradient re-parameterized optimizer that lets federated clients train a plain single-branch model while matching a multi-branch reference.

The reference architecture is a Constant-Scale Linear Addition (CSLA) block whose output sums scaled parallel branches, e.g. $\alpha_3 W_3 \ast x + \alpha_1 W_1 \ast x$. Rather than carry those extra branches, FedRepOpt trains one merged operator $W'$ and reproduces the multi-branch training dynamics with two rules: initialize $W' \leftarrow \alpha_3 W_3 + \alpha_1 W_1$, and during every update multiply the gradient by a constant "Grad Mult" derived from the branch scales. Because the multiplier is a fixed scalar per parameter group, each client runs an ordinary SGD step with no added communication or per-round overhead, which keeps it cheap in the federated setting.

$$
\begin{aligned}
w_k^{(i+1)} &= w_k^{(i)} - \eta\,(\alpha_3^2 + \alpha_1^2)\,\frac{\partial \mathcal{L}_k}{\partial w_k^{(i)}}
\end{aligned}
$$

where $w_k^{(i)}$ is a parameter of client $k$ at local iteration $i$, $\eta$ is the learning rate, $\mathcal{L}_k$ is the client's local loss, and $(\alpha_3^2 + \alpha_1^2)$ is the Grad Mult, the squared sum of the CSLA branch scales applied uniformly across the merged operator's parameters.

Reference: Kin Wai Lau, Yasar Abbas Ur Rehman, Pedro Porto Buarque de Gusmão, Lai-Man Po, Lan Ma, Yuyang Xie, "FedRepOpt: Gradient Re-parametrized Optimizers in Federated Learning", arXiv 2024. https://arxiv.org/abs/2409.15898

---
[Back to the Canon](../README.md)
