# ZOQO

Implements ZOQO, zeroth-order quantized optimization that keeps every quantity on the quantization grid.

ZOQO trains a quantized model without any full-precision gradient or update. It estimates only the sign of the gradient from two forward passes at quantized perturbations of the parameters, and applies a step whose size is itself snapped to the quantization grid. The perturbation is drawn from a discrete set of multiples of the quantization scale, and parameters are clamped to the representable range after every step, so weights, noise, and learning rate all remain valid quantization levels throughout training.

$$
\begin{aligned}
s &= \frac{R_{\max} - R_{\min}}{2^{b} - 1}, \qquad \eta_q = \max\!\left\{\left\lfloor \eta / s \right\rfloor,\, 1\right\} \cdot s \\
\theta_t^{+} &= \mathrm{clamp}\!\left(\theta_{t-1} + u_t\right), \qquad \theta_t^{-} = \mathrm{clamp}\!\left(\theta_{t-1} - u_t\right) \\
\mathrm{sign}(\hat{g}_t) &= \mathrm{sign}\!\left(\ell(\theta_t^{+}) - \ell(\theta_t^{-})\right) \cdot \mathrm{sign}(u_t) \\
\theta_t &= \mathrm{clamp}\!\left(\theta_{t-1} - \eta_q \cdot \mathrm{sign}(\hat{g}_t)\right)
\end{aligned}
$$

where $\theta$ are the quantized parameters, $\eta$ the requested learning rate, $\eta_q$ its grid-snapped value, $\ell$ the loss, $\hat{g}_t$ the estimated gradient, $s$ the quantization scale set by the bit budget $b$ and the parameter range $[R_{\min}, R_{\max}]$, and $u_t$ the quantized perturbation sampled coordinate-wise from $\mathcal{B} = \{-ms, (-m+1)s, \dots, 0, \dots, (m-1)s, ms\}$ with $m = \max\{\lfloor \mu / s \rfloor,\, 1\}$ for zeroth-order step size $\mu$, and $\mathrm{clamp}$ projects onto $[R_{\min}, R_{\max}]$.

Reference: Noga Bar, Raja Giryes, "ZOQO: Zero-Order Quantized Optimization", arXiv preprint 2025. https://arxiv.org/abs/2501.06736

---
[Back to the Canon](../README.md)
