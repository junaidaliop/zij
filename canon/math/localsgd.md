# Local SGD

Implements Local SGD, a communication-efficient distributed SGD that lets each worker run several local steps before averaging.

Each of $K$ workers keeps its own iterate $\theta_t^k$ and takes ordinary stochastic gradient steps independently. The workers communicate only at a sparse set of synchronization rounds $\mathcal{I}_T \subseteq [T]$, at which their parameters are replaced by the average across all workers. Between rounds the models drift apart for at most $H$ local steps, so the number of communication rounds can be cut to roughly $O(\sqrt{T})$ while keeping the convergence rate of parallel mini-batch SGD.

$$
\theta_{t+1}^k :=
\begin{aligned}
&\begin{cases}
\theta_t^k - \gamma_t\, g_t^k, & t+1 \notin \mathcal{I}_T,\\
\dfrac{1}{K}\displaystyle\sum_{j=1}^{K}\bigl(\theta_t^j - \gamma_t\, g_t^j\bigr), & t+1 \in \mathcal{I}_T,
\end{cases}
\end{aligned}
$$

where $\theta_t^k$ is worker $k$'s parameter at step $t$, $\gamma_t$ is the step size, $g_t^k = \nabla f_{i_t^k}(\theta_t^k)$ is the stochastic gradient with $i_t^k$ sampled uniformly at random from $[n]$, $K$ is the number of workers, and $\mathcal{I}_T$ is the set of synchronization steps whose gap (maximum spacing between consecutive elements) is at most $H$.

Reference: Sebastian U. Stich, "Local SGD Converges Fast and Communicates Little", ICLR 2019. https://arxiv.org/abs/1805.09767

---
[Back to the Canon](../README.md)
