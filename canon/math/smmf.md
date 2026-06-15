# SMMF

Implements SMMF, an Adam-style optimizer whose first and second moments are square-matricized and stored as rank-1 nonnegative factors.

The full moment matrices are the dominant memory cost of Adam. SMMF first reshapes each rank-$d$ gradient into a near-square matrix $\bar G_t$ (square-matricization chooses the factor dimensions $\hat n, \hat m$ with $\hat n \hat m = \prod_i n_i$ that minimize $|\hat n - \hat m|$), then keeps only a rank-1 factorization of each square moment. Unlike Adafactor, which factorizes a moment that was already reconstructed from stale factors before the gradient is added, SMMF first decompresses the previous factors back into full matrices, applies the current gradient, and only then recompresses; this preserves the new gradient information in both moments. Because the first moment can be negative, its sign pattern is stored separately in a 1-bit matrix $S_{M_t}$ and the nonnegative factorization is applied to $|M_t|$. The factors are formed by row and column sums (a rank-1 nonnegative matrix factorization), and decompression is their outer product.

$$
\begin{aligned}
\bar G_t &= \mathrm{Sqmat}(G_t) \\
\hat M_{t-1} &= S_{M_{t-1}} \odot \big(r_{M_{t-1}} \otimes c_{M_{t-1}}\big), \qquad \hat V_{t-1} = r_{V_{t-1}} \otimes c_{V_{t-1}} \\
M_t &= \beta_1 \hat M_{t-1} + (1-\beta_1)\, \bar G_t, \qquad V_t = \beta_2 \hat V_{t-1} + (1-\beta_2)\, \bar G_t^{\,2} \\
S_{M_t} &= \mathrm{sign}(M_t), \\
r_{M_t} &= |M_t|\,\mathbf{1}, \quad c_{M_t} = \frac{\mathbf{1}^{\top} |M_t|}{\mathbf{1}^{\top} |M_t|\, \mathbf{1}}, \qquad r_{V_t} = V_t\,\mathbf{1}, \quad c_{V_t} = \frac{\mathbf{1}^{\top} V_t}{\mathbf{1}^{\top} V_t\, \mathbf{1}} \\
\theta_t &= \theta_{t-1} - \eta_t\, \mathrm{Reshape}\!\left(\frac{M_t}{\sqrt{V_t}+\epsilon}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $G_t$ the rank-$d$ gradient, $\bar G_t$ its square-matricized form, $M_t,V_t$ the square first and second moments, $\hat M_{t-1},\hat V_{t-1}$ the moments reconstructed from the stored factors, $r,c$ the rank-1 row/column factors, $S_{M_t}$ the 1-bit sign matrix of the first moment, $\otimes$ the outer product, $\odot$ the elementwise product, $\mathbf{1}$ a vector of ones, $\beta_1,\beta_2$ the decay rates, $\eta_t$ the learning rate, and $\epsilon$ a stability constant; only $r_{M_t},c_{M_t},S_{M_t},r_{V_t},c_{V_t}$ are retained between steps, giving $O(\hat n + \hat m)$ moment storage plus the 1-bit sign matrix.

Reference: Kwangryeol Park, Seulki Lee, "SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization", arXiv 2024. https://arxiv.org/abs/2412.08894

---
[Back to the Canon](../README.md)
