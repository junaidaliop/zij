# TGF / TGFQS

Implements TGF / TGFQS (Two-Gate Fraxis and Two-Gate Free Quaternion Selection), gradient-free sequential optimizers that update a pair of single-qubit gates in a parameterized quantum circuit at once.

The single-gate predecessors fix the rest of the circuit and write the local energy of one gate as a quadratic form in its parameterization, then take the global minimizer in closed form. In Fraxis the gate is a $\pi$ reflection $R(n) = n_x X + n_y Y + n_z Z$ on the unit axis $n$ ($|n| = 1$), and the energy is $E(n) = n^{\top} \mathcal{R}\, n$ with $\mathcal{R}$ a $3\times 3$ real symmetric matrix built from circuit evaluations; the optimum is the eigenvector of $\mathcal{R}$ with the smallest eigenvalue. FQS replaces the axis by a unit quaternion $q = (q_0, q_1, q_2, q_3)$ spanning all of SU(2), giving $E(q) = q^{\top} M q$ with $M$ a $4\times 4$ real symmetric matrix and the same smallest-eigenvalue update.

TGF and TGFQS optimize two such gates simultaneously. Because each gate enters the expectation value quadratically, the joint local energy is biquadratic, i.e. an exact quartic cost in the two parameter vectors $(n_1, n_2)$ or $(q_1, q_2)$, which is minimized on the unit spheres with a classical optimizer.

$$
\begin{aligned}
E(n) &= n^{\top} \mathcal{R}\, n, \qquad n^{\star} = \arg\min_{|n|=1} E(n) = v_{\min}(\mathcal{R}), \\
E(q) &= q^{\top} M q, \qquad q^{\star} = \arg\min_{|q|=1} E(q) = v_{\min}(M), \\
E(x_1, x_2) &= \sum_{i,j,k,l} T_{ij,kl}\, (x_1)_i (x_1)_j (x_2)_k (x_2)_l, \qquad (x_1^{\star}, x_2^{\star}) = \arg\min_{|x_1|=|x_2|=1} E(x_1, x_2).
\end{aligned}
$$

where $n \in \mathbb{R}^3$ is a single-qubit rotation axis, $q \in \mathbb{R}^4$ a unit quaternion, $\mathcal{R}$ and $M$ the real symmetric local-cost matrices estimated from circuit measurements, $v_{\min}(\cdot)$ the eigenvector of smallest eigenvalue, $x_m \in \{n_m, q_m\}$ the parameterization of gate $m$, and $T$ the quartic coefficient tensor of the exact two-gate local cost that is minimized over the two unit spheres by a classical optimizer.

Reference: Joona V. Pankkonen, "Two-Gate Extensions of Free Axis and Free Quaternion Selection for Sequential Optimization of Parameterized Quantum Circuits", arXiv 2026. https://arxiv.org/abs/2603.25876

---
[Back to the Canon](../README.md)
