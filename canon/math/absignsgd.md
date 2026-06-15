# ABSignSGD

Implements ABSignSGD, block-coordinate sign descent for memory-efficient LLM fine-tuning.

ABSignSGD combines block-coordinate updates with stateless sign-based descent. At each step it selects one block of parameters and updates only those coordinates by the sign of their stochastic gradient, leaving all other blocks frozen. Discarding gradient magnitude and storing no moments keeps memory low, while updating a single block per step lets the backward pass terminate early. Block selection is arbitrary as long as each block is touched at least once every $B$ steps, which permits depth-biased schedules that update deeper layers more often. The data-parallel variant ABSignSGD-MV aggregates only one-bit block gradient signs across agents via majority vote.

$$
\begin{aligned}
x^{k+1}_{i_k} &= x^{k}_{i_k} - \alpha \cdot \mathrm{sign}\!\left(g_{i_k}(x^k)\right), \\
x^{k+1}_{i} &= x^{k}_{i} \quad \text{for all } i \neq i_k, \\
x^{k+1}_{i_k} &= x^{k}_{i_k} - \alpha \cdot \mathrm{sign}\!\left(\sum_{j=1}^{n} \mathrm{sign}\!\left(g^{j}_{i_k}(x^k)\right)\right) \quad \text{(ABSignSGD-MV)}.
\end{aligned}
$$

where $x$ are the parameters partitioned into $N$ disjoint blocks $\{\pi_1,\dots,\pi_N\}$, $i_k$ is the block selected at iteration $k$, $\alpha$ is the learning rate, $g_{i_k}(x^k)$ is the stochastic gradient of the active block, and $g^{j}_{i_k}(x^k)$ is that block gradient computed by agent $j$ among $n$ data-parallel agents.

Reference: Yijie Zhou, Shi Pu, "Arbitrary-Order Block SignSGD for Memory-Efficient LLM Fine-Tuning", ICLR 2026. https://openreview.net/forum?id=NQsdnYkCar

---
[Back to the Canon](../README.md)
