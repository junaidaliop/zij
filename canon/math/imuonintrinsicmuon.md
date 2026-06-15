# iMuon (Intrinsic Muon)

Implements iMuon (Intrinsic Muon), a manifold-aware generalization of Muon that solves the spectral-norm linear maximization oracle intrinsically on Riemannian matrix manifolds.

Euclidean Muon takes the step $X_{+} = X - \eta\,\tau\,\mathrm{Ortho}(M)$, where $M$ is the gradient (or momentum) and $\mathrm{Ortho}(M) = UV^\top$ is the polar factor obtained from the SVD $M = U\Sigma V^\top$. This is steepest descent under the spectral norm in flat space and does not respect constraints such as low rank, orthogonality, or positive definiteness.

iMuon replaces the ambient norm ball with a tangent-space linear maximization oracle measured in the Riemannian metric, then retracts back onto the manifold. With metric operator $G_x$ (so $g_x(\xi,\zeta) = \langle G_x\xi,\zeta\rangle$ and $\|\xi\|_x = \|G_x^{1/2}\xi\|_F$), the oracle admits a closed form: scale the gradient by $G_x^{1/2}$, orthogonalize the result, and scale back by $G_x^{-1/2}$. For the spectral norm this recovers a polar-factor (Muon-style) orthogonalization, now intrinsic to the manifold geometry. Concrete closed forms follow on the fixed-rank, SPD, Stiefel, and Grassmann manifolds using only Euclidean gradients from backpropagation.

$$
\begin{aligned}
\xi_t^{*} &= \arg\max_{\xi \in T_{x_t}\mathcal{M},\ \varphi(G_{x_t}^{1/2}\xi)\le\tau}\ g_{x_t}(\xi,\ \mathrm{grad}\,f) \\
H_t &= G_{x_t}^{1/2}\,\mathrm{grad}\,f(x_t) = U\,\mathrm{diag}(\sigma)\,V^\top \\
Z_t^{*} &= U\,\mathrm{diag}(z^{*})\,V^\top, \qquad z^{*} = \arg\max_{\varphi(z)\le\tau}\ \langle z,\sigma\rangle \\
\xi_t^{*} &= G_{x_t}^{-1/2}\,Z_t^{*} \\
x_{t+1} &= R_{x_t}\!\left(-\eta\,\xi_t^{*}\right)
\end{aligned}
$$

where $\mathcal{M}$ is the parameter manifold, $T_{x}\mathcal{M}$ its tangent space at $x$, $\mathrm{grad}\,f$ the Riemannian gradient, $G_x$ the (self-adjoint, positive-definite) metric operator, $\varphi$ any unitarily invariant norm, $\tau$ the norm bound, $\eta$ the step size, $R_x$ the retraction, and $U,\sigma,V$ the SVD factors of $H_t$. For the spectral norm $\varphi=\|\cdot\|_2$ the inner problem gives $z^{*}=\tau\mathbf{1}$, so $Z_t^{*}=\tau\,UV^\top$ is the polar factor and $\xi_t^{*}=\tau\,G_{x_t}^{-1/2}UV^\top$.

Reference: Yibang Li, Bihari Lal Pandey, Ravi Sah, Andi Han, Cyrus Mostajeran, Pratik Jawanpuria, Bamdev Mishra, "Intrinsic Muon: Spectral Optimization on Riemannian Matrix Manifolds", arXiv 2026. https://arxiv.org/abs/2605.09238

---
[Back to the Canon](../README.md)
