# LD-SGD

Implements LD-SGD, local decentralized SGD that interleaves several local SGD steps with periodic gossip averaging.

In decentralized training, $n$ workers each hold a local model and a private data shard, and they communicate only with neighbors through a mixing matrix instead of a central server. Fully decentralized SGD gossips after every gradient step, which is communication-heavy. LD-SGD reduces this cost by letting each worker run a block of local SGD updates and mixing the models only at the iterations in a communication set $\mathcal{I}_T$. Choosing $\mathcal{I}_T$ trades communication for accuracy: mixing every step recovers decentralized SGD, while sparser mixing approaches independent local training.

Stacking the workers' parameters into columns of $X_t \in \mathbb{R}^{d\times n}$ and their stochastic gradients into $G(X_t;\xi_t)$, one iteration is

$$
\begin{aligned}
x_{t+1/2}^{(k)} &= x_t^{(k)} - \eta\,\nabla F_k\!\bigl(x_t^{(k)};\xi_t^{(k)}\bigr) \\
x_{t+1}^{(k)} &= \sum_{l \in \mathcal{N}_k} w_{kl}\, x_{t+1/2}^{(l)} \\
X_{t+1} &= \bigl(X_t - \eta\, G(X_t;\xi_t)\bigr)\, W_t,
\qquad
W_t =
\begin{cases}
W & t \in \mathcal{I}_T \\
I_n & t \notin \mathcal{I}_T
\end{cases}
\end{aligned}
$$

where $x_t^{(k)}$ is worker $k$'s model, $\eta$ the learning rate, $\nabla F_k(\cdot;\xi^{(k)})$ its local stochastic gradient, $\mathcal{N}_k$ the neighbors of $k$, and $W=[w_{kl}]$ a symmetric doubly stochastic mixing matrix ($W=W^\top$, $W\mathbf{1}_n=\mathbf{1}_n$, $w_{kl}\ge 0$). At local iterations ($t \notin \mathcal{I}_T$) $W_t=I_n$, so no communication occurs and each worker takes a plain SGD step; at communication iterations ($t \in \mathcal{I}_T$) the half-step models are mixed through $W$.

Reference: Xiang Li, Wenhao Yang, Shusen Wang, Zhihua Zhang, "Communication-Efficient Local Decentralized SGD Methods", arXiv 2019. https://arxiv.org/abs/1910.09126

---
[Back to the Canon](../README.md)
