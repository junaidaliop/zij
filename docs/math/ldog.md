# LDoG

Implements LDoG, the layer-wise variant of DoG.

Applies the DoG step size formula to each parameter tensor (layer)
$\ell$ separately:


$$
\eta_t^{(\ell)} = \frac{\max_{i \le t} \lVert \theta_i^{(\ell)} -
\theta_0^{(\ell)} \rVert}{\sqrt{\sum_{i \le t} \lVert g_i^{(\ell)}
\rVert^2}}
$$


**Note:** Leave `lr` at its default of 1.0. The paper recommends pairing LDoG with polynomial decay iterate averaging.

Reference: Maor Ivgi, Oliver Hinder, Yair Carmon,
"DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule",
ICML 2023.
https://arxiv.org/abs/2302.12022

---
[Back to the Canon](../index.md)
