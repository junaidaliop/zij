# AdaCGD

Implements AdaCGD, communication-efficient distributed gradient descent with an adaptive three-point compressor.

AdaCGD trains a model across $n$ workers while compressing the gradients exchanged with the server. Compression is applied through a *three-point compressor* $\mathcal{C}_{h,y}(x)$ that depends not only on the vector $x$ being compressed but also on a reference point $h$ and an anchor $y$, which lets the contraction error shrink as the iterates approach the optimum. The descent step is plain gradient descent on the compressed aggregate gradient.

The adaptive part is the choice of compression intensity. At each step the operator $\mathcal{C}^{\mathrm{AC}}_{h,y}$ picks the cheapest of $m$ error-feedback compressors $\mathcal{C}^{(\mathrm{EF},j)}_{h,y}(x) = h + \mathcal{C}^{(j)}(x-h)$ whose output is close enough to $x$ relative to the anchor distance, falling back to sending $h$ when it already suffices and to the finest level $m$ otherwise.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \gamma\, g_t, \qquad g_t = \mathcal{C}^{\mathrm{AC}}_{h,y}\!\left(\frac{1}{n}\sum_{i=1}^{n} \nabla f_i(\theta_t)\right), \\
\mathcal{C}^{\mathrm{AC}}_{h,y}(x) &=
\begin{cases}
h, & \|x-h\|^2 \le \zeta\,\|x-y\|^2, \\
\mathcal{C}^{(\mathrm{EF},j)}_{h,y}(x), & \text{smallest } j<m \text{ with } \big\|x-\mathcal{C}^{(\mathrm{EF},j)}_{h,y}(x)\big\|^2 \le \zeta\,\|x-y\|^2, \\
\mathcal{C}^{(\mathrm{EF},m)}_{h,y}(x), & \text{otherwise},
\end{cases} \\
\mathcal{C}^{(\mathrm{EF},j)}_{h,y}(x) &= h + \mathcal{C}^{(j)}(x-h).
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the stepsize, $\nabla f_i$ the local gradient on worker $i$, $\mathcal{C}^{(j)}$ a contractive compressor at level $j$, $h$ the reference (learning) point and $y$ the anchor of the three-point compressor, and $\zeta \ge 0$ the threshold that triggers the switch between compression levels.

Reference: Maksim Makarenko, Elnur Gasanov, Rustem Islamov, Abdurakhmon Sadiev, Peter Richtárik, "Adaptive Compression for Communication-Efficient Distributed Training", arXiv 2022. https://arxiv.org/abs/2211.00188

---
[Back to the Canon](../README.md)
