# rTop-k

Implements rTop-k, a gradient sparsifier for distributed SGD that combines top-r selection with uniform random sampling.

In communication-efficient distributed training each worker compresses its gradient before transmission. Pure top-$k$ is deterministic and biased toward large coordinates, while random-$k$ is unbiased but ignores magnitude. rTop-k interpolates between the two: it first restricts attention to the $r$ coordinates of largest magnitude, then keeps a uniformly random $k$-subset of those, zeroing the rest. With error feedback, the coordinates dropped at one step are accumulated into a memory term and reintroduced at the next, so no gradient information is permanently discarded.

$$
\begin{aligned}
p_i^t &= g_i^t + m_i^t \\
\tilde{g}_i^t &= \mathrm{rTop}_k(p_i^t), \qquad
\big(\mathrm{rTop}_k(\omega)\big)_j =
\begin{cases} \omega_j & j \in U \\ 0 & j \notin U \end{cases} \\
m_i^{t+1} &= p_i^t - \tilde{g}_i^t \\
\theta_{t+1} &= \theta_t - \eta \cdot \frac{1}{n}\sum_{i=1}^{n} \tilde{g}_i^t
\end{aligned}
$$

where $g_i^t$ is the local stochastic gradient at worker $i$, $m_i^t$ is its accumulated error-feedback memory, $\theta$ are the model parameters, $\eta$ is the learning rate, and $n$ is the number of workers. The mask $U$ is drawn uniformly from the $k$-element subsets of the indices of the $r$ largest-magnitude coordinates of $p_i^t$ (with $1 \le k \le r \le d$), so $U$ is a random $k$-subset of the top-$r$ support.

Reference: Leighton Pate Barnes, Huseyin A. Inan, Berivan Isik, Ayfer Ozgur, "rTop-k: A Statistical Estimation Approach to Distributed SGD", arXiv 2020. https://arxiv.org/abs/2005.10761

---
[Back to the Canon](../README.md)
