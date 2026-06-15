# LOSCAR-SGD

Implements LOSCAR-SGD, local SGD that overlaps communication with computation and corrects the staleness through sparse model averaging.

Each round, every worker runs local SGD from its current iterate. It takes $N_i$ steps to produce a pre-communication model $y_i^r$, sparsifies it to $K$ coordinates, and ships it to the server while continuing to compute $Q_i$ overlap steps to reach $z_i^r$. The server averages the sparse messages into $\bar m^r$. Because each worker kept iterating past the snapshot it sent, naive overwriting would discard that overlap progress; instead the merge keeps the local iterate $z_i^r$ and adds, on the synchronized coordinates only, the correction $\bar m_j^r - y_{i,j}^r$, replacing each worker's stale contribution with the global average without throwing away its overlap work.

$$
\begin{aligned}
w_{i,0}^r &= x_i^r, \\
w_{i,t+1}^r &= w_{i,t}^r - \eta\, g_i(w_{i,t}^r, \xi_{i,t}^r), \quad t = 0,\dots,H_i-1, \\
y_i^r &= w_{i,N_i}^r, \qquad z_i^r = w_{i,H_i}^r, \\
\bar m^r &= \tfrac{1}{n} \sum_{i=1}^{n} \mathrm{Proj}_{S_r}\!\big(y_i^r\big), \\
x_{i,j}^{r+1} &=
\begin{cases}
\bar m_j^r + \big(z_{i,j}^r - y_{i,j}^r\big), & j \in S_r, \\
z_{i,j}^r, & j \notin S_r.
\end{cases}
\end{aligned}
$$

where $\theta$ is the model split across $n$ workers as iterates $w_{i,t}^r$, $\eta$ is the stepsize, $g_i$ the stochastic gradient with sample $\xi_{i,t}^r$, $H_i = N_i + Q_i$ the total local steps with $Q_i$ overlap steps, $S_r \subseteq [d]$ a uniformly random mask of $K$ coordinates shared by all workers in round $r$, $\mathrm{Proj}_{S_r}$ the projection that keeps those coordinates and zeros the rest, and $\bar m^r$ the server's sparse average over the $n$ workers.

Reference: Yassine Maziane, Ammar Mahran, Artavazd Maranjyan, Peter Richtárik, "LOSCAR-SGD: Local SGD with Communication-Computation Overlap and Delay-Corrected Sparse Model Averaging", arXiv 2025. https://arxiv.org/abs/2605.20866

---
[Back to the Canon](../README.md)
