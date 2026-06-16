# H-Fac

Implements H-Fac, a memory-efficient optimizer that factorizes both the momentum and the second moment into rank-1 row and column statistics derived from a Hamiltonian descent formulation.

For a matrix parameter $X \in \mathbb{R}^{m \times n}$, H-Fac avoids storing full $m \times n$ states. The first moment is tracked by two vectors $u_t \in \mathbb{R}^m$ (row averages of the gradient) and $v_t \in \mathbb{R}^n$ (column averages), and the second moment by two vectors $r_t \in \mathbb{R}^m$ and $s_t \in \mathbb{R}^n$, reconstructed as a rank-1 matrix in the spirit of Adafactor. The update is the discretization of a factorized Hamiltonian whose dissipation guarantees descent, combining a normalized factorized momentum term with an Adafactor-style update-clipped gradient term.

$$
\begin{aligned}
u_t &= \hat{\beta}_{1t}\, u_{t-1} + (1-\hat{\beta}_{1t})\, G_t 1_n / n \\
v_t &= \hat{\beta}_{1t}\, v_{t-1} + (1-\hat{\beta}_{1t})\, G_t^{\top} 1_m / m \\
r_t &= \hat{\beta}_{2t}\, r_{t-1} + (1-\hat{\beta}_{2t})\, \big[(G_t)^2 + \epsilon\big] 1_n \\
s_t &= \hat{\beta}_{2t}\, s_{t-1} + (1-\hat{\beta}_{2t})\, \big[(G_t^{\top})^2 + \epsilon\big] 1_m \\
\hat{V}_t &= r_t s_t^{\top} / (1_m^{\top} r_t) \\
\phi_t &= \hat{\beta}_{1t}\big(u_t 1_n^{\top} - G_t 1_n 1_n^{\top}/n\big) \big/ \sqrt{r_t 1_n^{\top}/n} \\
\psi_t &= \hat{\beta}_{1t}\big(1_m v_t^{\top} - 1_m 1_m^{\top} G_t/m\big) \big/ \sqrt{1_m s_t^{\top}/m} \\
X_t &= X_{t-1} - \eta_t\left( 0.5\,(\phi_t + \psi_t) + \mathrm{clip}\big(G_t / \sqrt{\hat{V}_t}\big) + \lambda X_{t-1} \right)
\end{aligned}
$$

where $G_t$ is the gradient at $X_{t-1}$, $1_m,1_n$ are all-ones vectors, $\hat{\beta}_{1t},\hat{\beta}_{2t}$ are the time-corrected decay rates (with $\hat{\beta}_{2t}=\beta_2(1-\beta_2^{t-1})/(1-\beta_2^{t})$), $\eta_t$ is the learning rate, $\lambda$ is the decoupled weight decay, $\epsilon$ is a stability constant, and $\mathrm{clip}(U)=U/\max(1,\mathrm{RMS}(U)/d)$ rescales by the root-mean-square with threshold $d$. Products and powers act elementwise, and division by $\sqrt{\hat{V}_t}$ is elementwise.

Reference: Son Nguyen, Lizhang Chen, Bo Liu, Qiang Liu, "Memory-Efficient Optimization with Factorized Hamiltonian Descent", arXiv 2024. https://arxiv.org/abs/2406.09958

---
[Back to the Canon](../index.md)
