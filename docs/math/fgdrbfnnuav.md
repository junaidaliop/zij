# FGD-RBFNN (UAV)

Implements FGD-RBFNN, a fractional gradient descent rule for training the output weights of a radial basis function neural network.

The RBFNN is used as an online identifier inside the active fault-tolerant controller of a plant-protection UAV. Each output weight is updated by a convex combination (mixing parameter $q$) of the ordinary integer-order gradient and a fractional-order ($\upsilon$) gradient of the squared-error cost $\varepsilon$. The fractional term, obtained from the Riemann–Liouville derivative of the cost with respect to the weight, contributes a $w_i^{1-\upsilon}$ factor scaled by $1/\Gamma(2-\upsilon)$, which lets the step adapt to the current weight magnitude.

$$
\begin{aligned}
-\nabla_{w_i}\varepsilon(n) &= \phi_i(x,x_i)\, e_k(n), \\
-\nabla^{\upsilon}_{w_i}\varepsilon(n) &= \phi_i(x,x_i)\, e_k(n)\, \frac{w_i^{\,1-\upsilon}(n)}{\Gamma(2-\upsilon)}, \\
w_i(n+1) &= w_i(n) + e_k(n)\,\Big( q\,\hbar + (1-q)\,\hbar_{\upsilon}\, w_i^{\,1-\upsilon}(n) \Big)\,\phi_i(x,x_i)
\end{aligned}
$$

where $w_i$ is the $i$-th output weight, $n$ the iteration index, $e_k(n)$ the instantaneous output error, $\phi_i(x,x_i)$ the $i$-th Gaussian basis activation, $\upsilon \in (0,1)$ the fractional order, $q \in [0,1]$ the convex mixing parameter, $\hbar$ and $\hbar_{\upsilon}$ the integer- and fractional-order step sizes, and $\Gamma(\cdot)$ the gamma function (folded into $\hbar_{\upsilon}$ in the combined form).

Reference: Lianghao Hua, Jianfeng Zhang, Dejie Li, Xiaobo Xi, "Fractional Gradient Descent RBFNN for Active Fault-Tolerant Control of Plant Protection UAVs", Computer Modeling in Engineering & Sciences 2024. https://doi.org/10.32604/cmes.2023.030535

---
[Back to the Canon](../index.md)
