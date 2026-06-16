# SFM

Implements SFM (Stochastic Fractional Moment Gradient Descent), a momentum optimizer that replaces the integer-order gradient with a fractional-order gradient.

SFM augments stochastic gradient descent with a fractional-calculus gradient component, so each parameter update aggregates gradient information across past iterates through a memory effect controlled by the fractional order rather than using only the instantaneous gradient. A momentum term is added on top of this fractional gradient to accelerate convergence and to help escape poor local minima. The method was introduced to train a U-Net for brain tumor segmentation in MRI, where it improved Dice score and convergence speed over Adam and SGD.

Reference: Anjali Malik, Ganesh Gopal Devarajan, "A momentum-based stochastic fractional gradient optimizer with U-net model for brain tumor segmentation in MRI", Digital Signal Processing 159, 2025. https://doi.org/10.1016/j.dsp.2025.104983

---
[Back to the Canon](../index.md)
