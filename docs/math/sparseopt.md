# SparseOpt

Implements SparseOpt, an SGD variant that corrects normalization-induced gradient skew in sparse training via a per-neuron diagonal preconditioner.

In sparse networks followed by batch normalization, surviving weights of heavily pruned neurons receive disproportionately amplified gradients, skewing updates toward the sparsest neurons. SparseOpt counteracts this by scaling each neuron's incoming-weight gradients by $\sqrt{1-s_i}$, where $s_i$ is that neuron's sparsity level, then rescaling by $1/\sqrt{1-s_{\mathrm{avg}}}$ to approximately preserve the global gradient norm. For a fully dense layer the preconditioner reduces to the identity and the method coincides with plain SGD.

$$
\begin{aligned}
D &= \mathrm{diag}\!\left(\sqrt{1-s_1},\,\sqrt{1-s_2},\,\dots,\,\sqrt{1-s_n}\right) \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{1-s_{\mathrm{avg}}}}\, D\, g_t
\end{aligned}
$$

where $\theta$ are the layer weights, $g_t = \nabla \mathcal{L}(\theta_t)$ the gradient, $\eta$ the learning rate, $s_i \in [0,1]$ the sparsity of neuron $i$, $D$ the diagonal preconditioner whose entry for neuron $i$ scales that neuron's incoming weights, and $s_{\mathrm{avg}}$ the average sparsity across the layer's neurons.

Reference: Mohammed Adnan, Rohan Jain, Tom Jacobs, Ekansh Sharma, Rahul G. Krishnan, Rebekka Burkholz, Yani Ioannou, "SparseOpt: Addressing Normalization-induced Gradient Skew in Sparse Training", ICML 2025. https://arxiv.org/abs/2605.27541

---
[Back to the Canon](../index.md)
