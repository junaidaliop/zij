# Freon / Kaon

Implements Freon / Kaon, Muon-family spectral optimizers that replace the orthogonalization step with a tunable Schatten-norm transform (Freon) or a chaotic singular-value scrambler (Kaon).

Both reuse Muon's outer loop: a momentum buffer $m_t = \mu m_{t-1} + g_t$ is formed, a matrix transform $O_t = f(m_t)$ is applied, and the parameter is stepped by $\theta_t = \theta_{t-1} - \eta\, O_t$. Freon generalizes Muon's polar factor $(G G^\top)^{-1/2} G$ to $(G G^\top)^{-c} G$ with $c = 1 - q/2$, interpolating between SGD ($c=0$) and Muon ($c=1/2$, i.e. $q=1$) and extrapolating into the quasi-norm regime $c>1/2$. Kaon discards the target spectrum entirely: it drives the gradient through a chaotic iteration whose multiplier sits deep in the logistic-map chaotic regime, scrambling the singular values yet still matching Muon's performance — the paper's central point that the precise output spectrum is largely irrelevant.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + g_t \\
\text{Freon:}\quad O_t &= \big(G_n G_n^\top\big)^{-c}\, G_n, \quad G_n = \frac{m_t}{\lVert m_t \rVert_q}, \quad c = 1 - \tfrac{q}{2} \\
\text{Kaon:}\quad X_0 &= \frac{m_t}{\lVert m_t \rVert_F}, \quad X_k = 4.1\,\big(I - X_{k-1} X_{k-1}^\top\big)^2 X_{k-1}, \quad O_t = \frac{X_T}{1.175} \\
\theta_t &= \theta_{t-1} - \eta\, O_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient (matrix-shaped), $m_t$ the momentum buffer with decay $\mu$ (0.95 in the paper), $\lVert\cdot\rVert_q$ the Schatten-$q$ norm, $\lVert\cdot\rVert_F$ the Frobenius norm, $c$ the Schatten exponent controlling Freon's spectral shaping, and $T$ the number of Kaon iterations (with constants $4.1$ and $1.175$ fixed by the chaotic map).

Reference: Zakhar Shumaylov, Nathaël Da Costa, Peter Zaika, Bálint Mucsányi, Alex Massucco, Yoav Gelberg, Carola-Bibiane Schönlieb, Yarin Gal, Philipp Hennig, "Muon is Not That Special: Random or Inverted Spectra Work Just as Well", arXiv 2026. https://arxiv.org/abs/2605.11181

---
[Back to the Canon](../index.md)
