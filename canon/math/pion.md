# Pion

Implements Pion, a spectrum-preserving optimizer that updates each weight matrix by orthogonal equivalence transformation.

Unlike additive optimizers such as Adam or Muon, which add a step to the weights, Pion multiplies each weight matrix on the left and right by orthogonal transformations. Because orthogonal factors leave singular values unchanged, the spectrum of every weight matrix is preserved exactly throughout training, removing the need for explicit normalization. The gradient is first projected onto the Lie algebra of skew-symmetric matrices on each side, accumulated with Adam-style first and second moments, and applied through a truncated matrix exponential.

$$
\begin{aligned}
G_t^{\mathrm{in}} &= W_t^\top g_t - g_t^\top W_t, \qquad G_t^{\mathrm{out}} = g_t W_t^\top - W_t g_t^\top \\
m_t^{\mathrm{in}} &= \beta_1 m_{t-1}^{\mathrm{in}} + (1-\beta_1) G_t^{\mathrm{in}}, \qquad v_t^{\mathrm{in}} = \beta_2 v_{t-1}^{\mathrm{in}} + (1-\beta_2) (G_t^{\mathrm{in}} \odot G_t^{\mathrm{in}}) \\
m_t^{\mathrm{out}} &= \beta_1 m_{t-1}^{\mathrm{out}} + (1-\beta_1) G_t^{\mathrm{out}}, \qquad v_t^{\mathrm{out}} = \beta_2 v_{t-1}^{\mathrm{out}} + (1-\beta_2) (G_t^{\mathrm{out}} \odot G_t^{\mathrm{out}}) \\
A_t^{\mathrm{in}} &= -\frac{m_t^{\mathrm{in}}}{\sqrt{v_t^{\mathrm{in}}}+\epsilon}, \qquad A_t^{\mathrm{out}} = -\frac{m_t^{\mathrm{out}}}{\sqrt{v_t^{\mathrm{out}}}+\epsilon} \\
\mathcal{E}_2(A,\alpha) &= I + \eta\alpha A + \tfrac{1}{2}(\eta\alpha A)^2, \qquad \alpha_t = \frac{c\sqrt{d_{\mathrm{out}} d_{\mathrm{in}}}}{\lVert A_t^{\mathrm{out}} W_{t-1} + W_{t-1} A_t^{\mathrm{in}} \rVert_F + \epsilon} \\
W_t &= \mathcal{E}_2(A_t^{\mathrm{out}}, \alpha_t)\, W_{t-1}\, \mathcal{E}_2(A_t^{\mathrm{in}}, \alpha_t)
\end{aligned}
$$

where $W_t$ is the weight matrix of shape $d_{\mathrm{out}} \times d_{\mathrm{in}}$, $g_t$ its gradient, $G_t^{\mathrm{in}}, G_t^{\mathrm{out}}$ the skew-symmetric input- and output-side projections, $m_t, v_t$ the first and second moments per side, $A_t$ the Adam-style adaptive directions, $\mathcal{E}_2$ the second-order truncation of the matrix exponential, $\eta$ the learning rate, $\alpha_t$ the RMS-based per-matrix scale, $c$ a target-RMS constant, $\beta_1, \beta_2$ the moment decay rates, and $\epsilon$ a stability constant.

Reference: Kexuan Shi, Hanxuan Li, Zeju Qiu, Yandong Wen, Simon Buchholz, Weiyang Liu, "Pion: A Spectrum-Preserving Optimizer via Orthogonal Equivalence Transformation", arXiv 2026. https://arxiv.org/abs/2605.12492

---
[Back to the Canon](../README.md)
