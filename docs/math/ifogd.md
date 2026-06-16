# IFOGD

Implements IFOGD (Improved Fractional-Order Gradient Descent), a Caputo fractional-order optimizer that extends fractional differentiation to a network's hidden layers.

Standard fractional-order gradient descent replaces the integer-order derivative of the loss with a Caputo fractional derivative, giving the update a tunable memory term controlled by the fractional order $\alpha$, but it can only be applied to the output layer. IFOGD reworks the Caputo-based symbolic differentiation formula so it splits into an integer-order-derivative component and a variable-dependent power function; the latter is recast as a tractable "fractional matrix differentiation" that is compatible with automatic differentiation (FOAutograd), which is what lets the fractional gradient propagate through hidden layers. The method also resolves the backpropagation-direction issue caused by the absolute-value sign in the fractional term, and is combined with first-order adaptive optimizers (Adam) to stabilize convergence.

The method generalizes the gradient step using fractional matrix differentiation, implemented through a fractional-order automatic-differentiation (FOAutograd) construction.

Reference: Xiaojun Zhou, Chunna Zhao, Yaqun Huang, Chengli Zhou, Junjie Ye, "Improved fractional-order gradient descent method based on multilayer perceptron", Neural Networks 183 (2025) 106970. https://doi.org/10.1016/j.neunet.2024.106970

---
[Back to the Canon](../index.md)
