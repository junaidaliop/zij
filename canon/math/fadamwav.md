# FAdamWav

Implements FAdamWav, a fractional version of Adam that runs its moment updates in a wavelet-compressed gradient space.

FAdamWav combines two ideas on top of Adam. First, each gradient entry is scaled by a Caputo-derived fractional factor: applying the Caputo derivative to $f(x)=x$ yields $x^{1-\nu}/\Gamma(2-\nu)$, so replacing the bare gradient with $g_t\,f_w^\nu$ generalizes the descent step to fractional order $\nu$ (recovering plain Adam at $\nu=1$). Second, the fractional gradient is passed through a parametric discrete wavelet transform (PDWT); only the low-frequency band $L_t$ is kept while the high-frequency band $H_t$ is zeroed, so the first and second moments are stored and updated on the compressed coefficients, saving 50%/75%/87.5% of moment memory for one/two/three transform levels. The Adam-style update is applied in the wavelet domain, then the inverse transform (PIDWT) maps the result back to the parameter shape for the weight step.

$$
\begin{aligned}
f_w^\nu &= \frac{\left(|\theta_{t-1}| + \epsilon\right)^{1-\nu}}{\Gamma(2-\nu)}, \qquad \tilde g_t = g_t \cdot f_w^\nu, \\
L_t &= \mathrm{PDWT}(\tilde g_t), \qquad H_t \leftarrow 0, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,L_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,L_t^2, \\
L_t &\leftarrow \frac{m_t}{\sqrt{v_t}+\epsilon}, \\
\hat g_t &= \mathrm{PIDWT}\!\left([L_t, H_t]\right), \\
\eta_t &= \eta \cdot \frac{1-\beta_2^{\,t}}{1-\beta_1^{\,t}}, \\
\theta_t &= \theta_{t-1} - \eta_t\,\hat g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the gradient, $\nu$ is the fractional order, $\Gamma(\cdot)$ is the gamma function, $f_w^\nu$ is the fractional gradient factor, $\mathrm{PDWT}/\mathrm{PIDWT}$ are the parametric forward/inverse discrete wavelet transforms, $L_t$ and $H_t$ are the low- and high-frequency wavelet sub-bands (with $H_t$ erased for memory reduction), $m_t,v_t$ are the first and second moments computed in the wavelet domain, $\beta_1,\beta_2$ are the decay rates, $\eta_t$ is the bias-correction-scaled step, and $\epsilon>0$ guards against division by zero and weight-magnitude indeterminacy.

Reference: Oscar Herrera-Alcántara, Salvador Arellano-Balderas, Sandra Rodríguez-Mondragón, José Alejandro Reyes-Ortíz, Jaime Navarro-Fuentes, "FAdamWav: A Fractional Wavelet Gradient Optimizer for Neural Networks", Fractal and Fractional 2026. https://doi.org/10.3390/fractalfract10030149

---
[Back to the Canon](../README.md)
