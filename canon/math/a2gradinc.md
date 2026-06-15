# A2GradInc

Implements A2Grad (incremental variant), adaptive accelerated SGD.

Same accelerated coupling as `A2GradUni`, but the adaptive
accumulator is discounted incrementally before each gradient deviation
is added:


$$
v_k = \left(\frac{k}{k + 1}\right)^2 v_{k-1}
      + \lVert \delta_k \rVert^2
$$

Reference: Qi Deng, Yi Cheng, Guanghui Lan, "Optimal Adaptive and
Accelerated Stochastic Gradient Descent", arXiv 2018.
https://arxiv.org/abs/1810.00553

---
[Back to the Canon](../README.md)
