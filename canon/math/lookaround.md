# Lookaround

Implements Lookaround, a weight-averaging optimizer that runs $d$ diverse copies for $k$ steps, then averages them.

Lookaround wraps an inner optimizer $A$ (e.g. SGD). It keeps a single set of "around" weights and, each cycle, branches them into $d$ copies that train independently for $k$ steps, each copy fed a differently augmented view of the same minibatches. After the $k$ around steps, the copies are arithmetically averaged into one set of weights, which seeds the next cycle. This alternation of $k$ divergent steps and one averaging step balances local exploration against the flat-minimum benefit of weight averaging.

$$
\begin{aligned}
\theta_{t,j,0} &\leftarrow \phi_{t-1}, \quad j = 1,\dots,d \\
\theta_{t,j,i} &\leftarrow \theta_{t,j,i-1} + A\big(\mathcal{L}, \theta_{t,j,i-1}, \mathrm{AUG}_j(B)\big), \quad i = 1,\dots,k \\
\phi_t &\leftarrow \frac{1}{d}\sum_{j=1}^{d} \theta_{t,j,k}
\end{aligned}
$$

where $\phi_{t-1}$ are the around weights at the start of cycle $t$, $\theta_{t,j,i}$ are the weights of copy $j$ after $i$ around steps, $d$ is the number of copies, $k$ is the synchronization period, $A$ is the inner optimizer applied to loss $\mathcal{L}$, $B$ is a sampled minibatch, and $\mathrm{AUG}_j$ is the data augmentation assigned to copy $j$.

Reference: Jiangtao Zhang, Shunyu Liu, Jie Song, Tongtian Zhu, Zhengqi Xu, Mingli Song, "Lookaround Optimizer: $k$ steps around, 1 step average", NeurIPS 2023. https://arxiv.org/abs/2306.07684

---
[Back to the Canon](../README.md)
