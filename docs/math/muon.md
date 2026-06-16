# Muon

Implements Muon, a momentum optimizer for 2D hidden-layer weights that
orthogonalizes each update with a Newton-Schulz iteration.

Muon takes the standard SGD-momentum buffer $m_t$ and replaces it with the
nearest semi-orthogonal matrix before stepping. The orthogonalization is done
by a fixed-coefficient Newton-Schulz iteration that, starting from the
Frobenius-normalized update, drives the singular values toward one without an
explicit SVD. Equalizing the singular values prevents a few dominant directions
from steering the step, so every direction in the weight matrix receives a
comparable update. Scalar and vector parameters, along with the input and
output layers, are left to a method such as AdamW.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + g_t \\
X_0 &= m_t / \lVert m_t \rVert_F \\
X_{k+1} &= a X_k + b (X_k X_k^{\top}) X_k + c (X_k X_k^{\top})^2 X_k,
\qquad k = 0, \dots, N-1 \\
O_t &= X_N \\
\theta_t &= \theta_{t-1} - \eta\, O_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the
gradient, $m_t$ is the momentum buffer, $\mu$ is the momentum coefficient,
$\lVert \cdot \rVert_F$ is the Frobenius norm, $O_t$ is the orthogonalized
update, and $(a, b, c) = (3.4445, -4.7750, 2.0315)$ are the fixed Newton-Schulz
coefficients run for $N = 5$ iterations.

Reference: Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista,
Laker Newhouse, Jeremy Bernstein, "Muon: An optimizer for hidden layers in
neural networks", Blog 2024.
https://kellerjordan.github.io/posts/muon/

---
[Back to the Canon](../index.md)
