# FracG

Implements FracG, a fractional-order gradient method that rectifies the gradient with a Caputo-type fractional derivative.

FracG replaces the integer-order gradient in the parameter update with a fractional-order gradient of order $\alpha$, exploiting the power-law memory of the Caputo derivative so that each step retains information from past states. This memory effect is intended to help the optimizer escape poor local extrema when training deep networks on large, complex data, while preserving stable and generalizable weight updates.

Reference: Zhongliang Yu, Jianfeng Lv, Erwei Li, "Optimization Method of Neural Networks via Fractional-Order of Gradients", Chinese Control Conference (CCC) 2023. https://doi.org/10.23919/CCC58697.2023.10239893

---
[Back to the Canon](../index.md)
