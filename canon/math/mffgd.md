# MFFGD

Implements MFFGD, an adaptive Caputo fractional-order gradient method for deep networks.

MFFGD replaces the integer-order gradient with a simplified Caputo fractional-order derivative of the loss, whose order $\alpha \in (0,1)$ controls how much past curvature is folded into each step. The authors derive the fractional-order gradient and its error analysis for common activation and loss functions, which simplifies the otherwise expensive evaluation of the Caputo derivative during backpropagation.

The adaptive part is a memory factor that records past gradient variations and uses them to adjust the learning rate online, by effectively reshaping the integration domain of the fractional derivative. This reduces the hyperparameter count and is reported to improve generalization and resistance to poor local optima relative to standard gradient descent.

Reference: Zhuo Huang, Shuhua Mao, Yingjie Yang, "MFFGD: An adaptive Caputo fractional-order gradient algorithm for DNN", Neurocomputing 2024. https://doi.org/10.1016/j.neucom.2024.128606

---
[Back to the Canon](../README.md)
