# LoRA-RITE

Implements LoRA-RITE, a transformation-invariant adaptive optimizer for low-rank (LoRA) adapters.

LoRA parameterizes a weight update as $Z = AB^\top$ with factors $A$ and $B$. Standard adaptive optimizers depend on the particular factorization chosen, so two equivalent factorizations of the same $Z$ produce different updates. LoRA-RITE removes this dependence by stripping each factor's magnitude through a polar decomposition $A = U_A R_A$, $B = U_B R_B$ (orthonormal $U$, upper-triangular $R$) and preconditioning the resulting "unmagnified" gradients in the shared column space. Because the basis $U_B$ rotates between steps, the second-moment accumulator is transported by the projection $P_{A,t} = U_{B,t}^\top U_{B,t-1}$, and a scalar $\rho$ compensates for the spectral mass lost under that projection. A final right-multiplication restores the correct magnitude, yielding an update on $Z$ that is invariant to the choice of factorization. The $B$ factor is updated symmetrically with $A$ and $B$ roles swapped.

$$
\begin{aligned}
\bar{g}_{A,t} &= (\nabla_A)_t\, R_{B,t}^{-1} \\
P_{A,t} &= U_{B,t}^\top U_{B,t-1} \\
\bar{V}_{A,t} &= P_{A,t}\,\bar{V}_{A,t-1}\,P_{A,t}^\top + \bar{g}_{A,t}^\top \bar{g}_{A,t} \\
\rho_{A,t} &= \rho_{A,t-1} + d_\lambda\!\big(\bar{V}_{A,t-1},\; P_{A,t}\,\bar{V}_{A,t-1}\,P_{A,t}^\top\big) \\
\bar{S}_{A,t} &= \bar{g}_{A,t}\,\big(\bar{V}_{A,t} + \rho_{A,t} I\big)^{-1/2} \\
\bar{M}_{A,t} &= \beta_1\,\bar{M}_{A,t-1}\,P_{A,t}^\top + (1-\beta_1)\,\bar{S}_{A,t} \\
A_{t+1} &= A_t - \eta_t\,\bar{M}_{A,t}\,R_{B,t}^{-\top}
\end{aligned}
$$

where $A = U_A R_A$ and $B = U_B R_B$ are polar decompositions of the LoRA factors, $\bar{g}_{A,t}$ is the magnitude-invariant gradient, $P_{A,t}$ transports state across the rotated basis $U_B$, $\bar{V}_{A,t}$ is the second moment, $\rho_{A,t}$ accumulates the spectral distance $d_\lambda$ of mass escaped by projection, $\bar{S}_{A,t}$ is the preconditioned direction, $\bar{M}_{A,t}$ the first moment with decay $\beta_1$, $\eta_t$ the learning rate, and $R_{B,t}^{-\top}$ restores magnitude.

Reference: Jui-Nan Yen, Si Si, Zhao Meng, Felix Yu, Sai Surya Duvvuri, Inderjit S. Dhillon, Cho-Jui Hsieh, Sanjiv Kumar, "LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization", arXiv 2024. https://arxiv.org/abs/2410.20625

---
[Back to the Canon](../index.md)
