# EF21-Muon

Implements EF21-Muon, a communication-efficient distributed Muon with EF21 error feedback on both gradient and model updates.

Muon takes a steepest-descent step under the spectral norm: the linear minimization oracle over a spectral-norm ball orthogonalizes the aggregated direction, giving $-U V^\top$ from its SVD (approximated in practice by Newton-Schulz iterations). EF21-Muon makes this scale across $n$ workers by applying EF21-style error feedback to the messages. Each worker maintains a momentum estimate and a gradient state $G_{j}$ that is updated only by a compressed correction $R_j$; the server keeps an aggregate $G$ and a compressed model shift $S$ that is broadcast back, so both worker-to-server and server-to-worker traffic is compressed while the error-feedback states $G_j$ and $W$ track the true quantities.

$$
\begin{aligned}
X^{k+1} &= \mathrm{LMO}_{\mathcal{B}(X^k,\,t^k)}(G^k), \qquad \mathrm{LMO}_{\mathcal{B}(0,1)}(G)= -U V^\top \\
S^k &= \mathcal{C}^k\!\left(X^{k+1}-W^k\right), \qquad W^{k+1}=W^k+S^k \\
M_j^{k+1} &= (1-\beta)\,M_j^k + \beta\,\nabla f_j\!\left(W^{k+1};\xi_j^{k+1}\right) \\
R_j^{k+1} &= \mathcal{C}_j^k\!\left(M_j^{k+1}-G_j^k\right), \qquad G_j^{k+1}=G_j^k+R_j^{k+1} \\
G^{k+1} &= G^k + \frac{1}{n}\sum_{j=1}^{n} R_j^{k+1}
\end{aligned}
$$

where $X$ is the server model, $W$ the compressed broadcast copy, $t^k$ the step size, $G$ the aggregated gradient estimate and $G_j$ its per-worker error-feedback state, $M_j$ worker momentum with decay $\beta$, $\nabla f_j(\cdot;\xi)$ a stochastic gradient, $\mathcal{C}^k$ and $\mathcal{C}_j^k$ contractive compressors satisfying $\mathbb{E}\lVert \mathcal{C}(x)-x\rVert^2 \le (1-\alpha)\lVert x\rVert^2$, $U V^\top$ comes from the SVD $G=U\Sigma V^\top$, and $n$ is the number of workers.

Reference: Kaja Gruntkowska, Alexander Gaponov, Zhirayr Tovmasyan, Peter Richtárik, "Error Feedback for Muon and Friends", arXiv 2025. https://arxiv.org/abs/2510.00643

---
[Back to the Canon](../README.md)
