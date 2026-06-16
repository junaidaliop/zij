# NCFDD / NFLightGBM

Implements NCFDD (non-causal fractional difference descent), a fractional-order gradient descent that replaces the integer derivative with a non-causal Grünwald–Letnikov fractional difference, and the NFLightGBM model that uses it to fit gradient-boosted trees.

The idea is that a one-sided (causal) fractional derivative introduces a phase shift that biases the search away from the true extremum. The paper builds a causal and an anti-causal Grünwald–Letnikov fractional difference and combines them as a weighted sum, yielding a non-causal fractional derivative whose phase spectrum matches the ordinary first derivative. This combined operator is substituted for the gradient in steepest descent, giving fast convergence with the global-extremum guarantee that pure fractional descent lacks. Plugging the resulting operator into LightGBM's gradient and Hessian computation produces NFLightGBM.

$$
\begin{aligned}
D_{+}^{\alpha} f(\theta) &= \lim_{h\to 0^{+}} \frac{1}{h^{\alpha}} \sum_{k=0}^{\infty} (-1)^{k}\binom{\alpha}{k} f(\theta - k h), \\
D_{-}^{\alpha} f(\theta) &= \lim_{h\to 0^{+}} \frac{1}{h^{\alpha}} \sum_{k=0}^{\infty} (-1)^{k}\binom{\alpha}{k} f(\theta + k h), \\
D^{\alpha} f(\theta) &= \lambda\, D_{+}^{\alpha} f(\theta) + (1-\lambda)\, D_{-}^{\alpha} f(\theta), \\
\theta_{t+1} &= \theta_{t} - \eta\, D^{\alpha} f(\theta_{t}).
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\alpha$ the fractional order, $h$ the step size, $\binom{\alpha}{k}$ the generalized binomial coefficient, $D_{+}^{\alpha}$ / $D_{-}^{\alpha}$ the causal and anti-causal Grünwald–Letnikov fractional derivatives, and $\lambda\in[0,1]$ the weight blending them into the non-causal derivative $D^{\alpha}$.

Reference: Haixin Wu, Yaqian Mao, Jiacheng Weng, Yue Yu, Jianhong Wang, "Fractional light gradient boosting machine ensemble learning model: A non-causal fractional difference descent approach", Information Fusion 118 (2025), 102947. https://doi.org/10.1016/j.inffus.2025.102947

---
[Back to the Canon](../index.md)
