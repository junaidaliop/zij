# Quantum and Quantum-Inspired Optimizers

This page collects optimizers from two adjacent settings. The first is the optimization of variational quantum circuits, where shot noise and the quantum geometry of the parameter space drive the design of measurement-frugal, gradient-free, and natural-gradient methods. The second is quantum-inspired and quantum-hardware optimization of classical neural networks, where quantum fluctuations, adiabatic evolution, or annealer sampling replace or augment the classical training loop.

## Optimizers for variational quantum circuits

| Optimizer | Venue | Paper | Code |
|---|---|---|---|
| [SPSA](math/spsa.md) | IEEE Transactions on Automatic Control 1992 | [Multivariate stochastic approximation using a simultaneous perturbation gradient approximation](https://doi.org/10.1109/9.119632) | [official](https://www.jhuapl.edu/spsa/) |
| [iCANS](math/icans.md) | Quantum 2020 | [An Adaptive Optimizer for Measurement-Frugal Variational Algorithms](https://quantum-journal.org/papers/q-2020-05-11-263/) | [community](https://docs.pennylane.ai/en/stable/code/api/pennylane.ShotAdaptiveOptimizer.html) |
| [NFT](math/nft.md) | Physical Review Research 2020 | [Sequential minimal optimization for quantum-classical hybrid algorithms](https://link.aps.org/doi/10.1103/PhysRevResearch.2.043158) | [official](https://gist.github.com/ken-nakanishi/e38de385b39017b6f673324a96ca16bd) |
| [Quantum Natural Gradient](math/quantumnaturalgradient.md) | Quantum 2020 | [Quantum Natural Gradient](https://quantum-journal.org/papers/q-2020-05-25-269/) | [official](https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/optimize/qng.py) |
| [Rosalin](math/rosalin.md) | arXiv 2020 | [Operator Sampling for Shot-frugal Optimization in Variational Algorithms](https://arxiv.org/abs/2004.06252) | [community](https://docs.pennylane.ai/en/stable/code/api/pennylane.ShotAdaptiveOptimizer.html) |
| [QN-SPSA](math/qnspsa.md) | Quantum 2021 | [Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information](https://quantum-journal.org/papers/q-2021-10-20-567/) | [official](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.QNSPSA.html) |
| [Rotosolve / Rotoselect](math/rotosolverotoselect.md) | Quantum 2021 | [Structure optimization for parameterized quantum circuits](https://quantum-journal.org/papers/q-2021-01-28-391/) | [community](https://docs.pennylane.ai/en/stable/code/api/pennylane.RotosolveOptimizer.html) |
| [Quantum Analytic Descent](math/quantumanalyticdescent.md) | Physical Review Research 2022 | [Quantum Analytic Descent](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.023017) | [official](https://github.com/BalintKoczor/quantum-analytic-descent) |
| [SGLBO](math/sglbo.md) | npj Quantum Information 2022 | [Stochastic gradient line Bayesian optimization for efficient noise-robust optimization of parameterized quantum circuits](https://www.nature.com/articles/s41534-022-00592-6) | [community](https://github.com/wntamanda/sglbo-quantum-opt) |
| [SantaQlaus](math/santaqlaus.md) | arXiv 2023 | [SantaQlaus: A resource-efficient method to leverage quantum shot-noise for optimization of variational quantum algorithms](https://arxiv.org/abs/2312.15791) | — |
| [ExcitationSolve](math/excitationsolve.md) | Communications Physics 2025 | [Fast gradient-free optimization of excitations in variational quantum eigensolvers](https://www.nature.com/articles/s42005-025-02375-9) | [official](https://github.com/dlr-wf/ExcitationSolve) |
| [Kernel Descent](math/kerneldescent.md) | Scientific Reports 2025 | [Introducing the kernel descent optimizer for variational quantum algorithms](https://www.nature.com/articles/s41598-025-08392-6) | — |
| [QUIVER](math/quiver.md) | arXiv 2026 | [Adaptive directional gradients for parameterised quantum circuits](https://arxiv.org/abs/2606.09734) | — | — |
| [WSBD](math/wsbd.md) | AISTATS 2026 | [WSBD: Freezing-Based Optimizer for Quantum Neural Networks](https://arxiv.org/abs/2602.11383) | [official](https://github.com/Damrl-lab/WSBD-Stochastic-Freezing-Optimizer) | — |
| [H-QNG](math/hqng.md) | arXiv 2025 | [Efficient Hamiltonian-aware Quantum Natural Gradient Descent for Variational Quantum Eigensolvers](https://arxiv.org/abs/2511.14511) | — | — |
| [WA-QNG](math/waqng.md) | Quantum Science and Technology 2026 | [Weighted Approximate Quantum Natural Gradient for Variational Quantum Eigensolver](https://arxiv.org/abs/2504.04932) | — | — |
| [CQNG](math/cqng.md) | EPJ Quantum Technology 2025 | [Modified Conjugate Quantum Natural Gradient](https://arxiv.org/abs/2501.05847) | — | — |
| [Momentum-QNG](math/momentumqng.md) | Physica A 2024 | [Application of Langevin Dynamics to Advance the Quantum Natural Gradient Optimization Algorithm](https://arxiv.org/abs/2409.01978) | [official](https://github.com/borbysh/Momentum-QNG) | — |
| [qBang](math/qbang.md) | Quantum 2024 | [Optimizing Variational Quantum Algorithms with qBang: Efficiently Interweaving Metric and Momentum to Navigate Flat Energy Landscapes](https://arxiv.org/abs/2304.13882) | [official](https://github.com/davidfitzek/qbang) | — |
| [EGT (Exact Geodesic Transport)](math/egtexactgeodesictransport.md) | arXiv 2025 | [Quantum optimization with exact geodesic transport](https://arxiv.org/abs/2506.17395) | — | — |
| [TGF / TGFQS](math/tgftgfqs.md) | arXiv 2026 | [Two-Gate Extensions of Free Axis and Free Quaternion Selection for Sequential Optimization of Parameterized Quantum Circuits](https://arxiv.org/abs/2603.25876) | — | — |
| [SGD (Superpositional Gradient Descent)](math/sgdsuperpositionalgradientdescent.md) | IEEE QAI 2025 | [Superpositional Gradient Descent: Harnessing Quantum Principles for Model Training](https://arxiv.org/abs/2511.01918) | — | — |
| [Scalable On-Hardware QNN training (parallelised parameter-shift rule)](math/scalableonhardwareqnntrainingparallelisedparametershiftrule.md) | arXiv 2026 | [Scalable On-Hardware Training of Quantum Neural Networks and Application to Clinical Data Imputation](https://arxiv.org/abs/2606.03517) | — | — |
| [QM-quantization optimizer (Schrodinger gradient-flow)](math/qmquantizationoptimizerschrodingergradientflow.md) | arXiv 2026 | [Quantum mechanical framework for quantization-based optimization: from Gradient flow to Schroedinger equation](https://arxiv.org/abs/2603.11536) | — | — |

## Quantum-inspired and quantum-hardware methods

| Optimizer | Venue | Paper | Code |
|---|---|---|---|
| [Quantum Adam](math/quantumadam.md) | Scientific Reports 2018 | [Optimization of neural networks via finite-value quantum fluctuations](https://www.nature.com/articles/s41598-018-28212-4) | — |
| [RBM training on a D-Wave annealer](math/rbmtrainingonadwaveannealer.md) | Frontiers in Physics 2021 | [Training Restricted Boltzmann Machines With a D-Wave Quantum Annealer](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.589626/full) | — |
| [Quantum Hamiltonian Descent (QHD)](math/quantumhamiltoniandescentqhd.md) | arXiv 2023 | [Quantum Hamiltonian Descent](https://arxiv.org/abs/2303.01471) | [official](https://github.com/jiaqileng/quantum-hamiltonian-descent) |
| [Universal AQC neural-network training](math/universalaqcneuralnetworktraining.md) | Frontiers in Artificial Intelligence 2024 | [Training neural networks with universal adiabatic quantum computing](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1368569/full) | — |
| [QHDOPT](math/qhdopt.md) | INFORMS Journal on Computing 2025 | [QHDOPT: A Software for Nonlinear Optimization with Quantum Hamiltonian Descent](https://pubsonline.informs.org/doi/10.1287/ijoc.2024.0587) | [official](https://github.com/PhysOpt/QHDOPT) |
| [Stochastic Quantum Hamiltonian Descent (SQHD)](math/stochasticquantumhamiltoniandescentsqhd.md) | arXiv 2025 | [Stochastic Quantum Hamiltonian Descent](https://arxiv.org/abs/2507.15424) | — |
| [QIASO](math/qiaso.md) | AIMS Mathematics 2026 | [The quantum-inspired adaptive superposition optimization for neural network training](https://www.aimspress.com/article/doi/10.3934/math.2026010) | — |
