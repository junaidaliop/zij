# Privacy-Preserving Optimizers

Privacy-preserving optimizers train models under differential privacy, typically by clipping per-sample gradients and adding calibrated noise to updates. This page lists differentially private optimization methods and reference libraries, from the original DP-SGD to later variants that reduce clipping bias, correct moment estimates, or filter privacy noise.

| Optimizer | Venue | Paper | Code |
|---|---|---|---|
| [DP-SGD](math/dpsgd.md) | CCS 2016 | [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) | [official](https://github.com/tensorflow/privacy) |
| [DP-LSSGD](math/dplssgd.md) | MSML 2020 | [DP-LSSGD: A Stochastic Optimization Method to Lift the Utility in Privacy-Preserving ERM](https://arxiv.org/abs/1906.12056) | [official](https://github.com/BaoWangMath/DP-LSSGD) |
| [DP-PASGD](math/dppasgd.md) | arXiv 2020 | [Differentially Private Federated Learning for Resource-Constrained Internet of Things](https://arxiv.org/abs/2003.12705) | — |
| [DP-SGD-JL](math/dpsgdjl.md) | NeurIPS 2021 | [Fast and Memory Efficient Differentially Private-SGD via JL Projections](https://arxiv.org/abs/2102.03013) | — |
| [Opacus](math/opacus.md) | arXiv 2021 | [Opacus: User-Friendly Differential Privacy Library in PyTorch](https://arxiv.org/abs/2109.12298) | [official](https://github.com/meta-pytorch/opacus) |
| [A(DP)²SGD](math/adpsgd.md) | TPAMI 2022 | [A(DP)²SGD: Asynchronous Decentralized Parallel Stochastic Gradient Descent with Differential Privacy](https://arxiv.org/abs/2008.09246) | — |
| [DPIS](math/dpis.md) | CCS 2022 | [DPIS: An Enhanced Mechanism for Differentially Private SGD with Importance Sampling](https://arxiv.org/abs/2210.09634) | — |
| [Top-DP](math/topdp.md) | TCSVT 2022 | [Topology-aware Differential Privacy for Decentralized Image Classification](https://arxiv.org/abs/2006.07817) | — |
| [ANSGD](math/ansgd.md) | arXiv 2023 | [Learning across Data Owners with Joint Differential Privacy](https://arxiv.org/abs/2305.15723) | — |
| [DP-FedSAM](math/dpfedsam.md) | CVPR 2023 | [Make Landscape Flatter in Differentially Private Federated Learning](https://arxiv.org/abs/2303.11242) | [official](https://github.com/YMJS-Irfan/DP-FedSAM) |
| [AClipped-dpSGD](math/aclippeddpsgd.md) | Machine Learning 2024 | [Efficient Private SCO for Heavy-Tailed Data via Averaged Clipping](https://arxiv.org/abs/2206.13011) | — |
| [DiceSGD](math/dicesgd.md) | ICLR 2024 | [Differentially Private SGD Without Clipping Bias: An Error-Feedback Approach](https://arxiv.org/abs/2311.14632) | [official](https://github.com/564612540/DiceSGD) |
| [DOPPLER](math/doppler.md) | NeurIPS 2024 | [DOPPLER: Differentially Private Optimizers with Low-pass Filter for Privacy Noise Reduction](https://arxiv.org/abs/2408.13460) | — |
| [DP-AdamBC](math/dpadambc.md) | AAAI 2024 | [DP-AdamBC: Your DP-Adam Is Actually DP-SGD (Unless You Apply Bias Correction)](https://arxiv.org/abs/2312.14334) | [official](https://github.com/ubc-systopia/DP-AdamBC) |
| [FedLAP-DP](math/fedlapdp.md) | PoPETs 2024 | [FedLAP-DP: Federated Learning by Sharing Differentially Private Loss Approximations](https://arxiv.org/abs/2302.01068) | [official](https://github.com/hui-po-wang/FedLAP-DP) |
| [DC-SGD](math/dcsgd.md) | TIFS 2025 | [DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation](https://arxiv.org/abs/2503.22988) | — |
| [DP-AdamW](math/dpadamw.md) | ICML Workshop 2025 | [DP-AdamW: Investigating Decoupled Weight Decay and Bias Correction in Private Deep Learning](https://arxiv.org/abs/2511.07843) | — |
| [DP-MicroAdam](math/dpmicroadam.md) | arXiv 2025 | [DP-MicroAdam: Private and Frugal Algorithm for Training and Fine-tuning](https://arxiv.org/abs/2511.20509) | — |
| [DPZV](math/dpzv.md) | arXiv 2025 | [Communication-Efficient and Differentially Private Vertical Federated Learning with Zeroth-Order Optimization](https://arxiv.org/abs/2502.20565) | — |
| [GeoDP](math/geodp.md) | ICDE 2025 | [Analyzing and Optimizing Perturbation of DP-SGD Geometrically](https://arxiv.org/abs/2504.05618) | [official](https://github.com/Derek0205/GeoDP) |
| [Interleaved-ShuffleG](math/interleavedshuffleg.md) | arXiv 2025 | [Improving the Convergence of Private Shuffled Gradient Methods with Public Data](https://arxiv.org/abs/2502.03652) | — |
| [Logit-DP](math/logitdp.md) | ICLR 2025 | [Differentially Private Optimization for Non-Decomposable Objective Functions](https://arxiv.org/abs/2310.03104) | — |
| [SPARTA](math/sparta.md) | KDD 2025 | [SPARTA: An Optimization Framework for Differentially Private Sparse Fine-Tuning](https://arxiv.org/abs/2503.12822) | [official](https://github.com/mazumder-lab/SPARTA) |
| [DP-λCGD](math/dpcgd.md) | arXiv 2026 | [DP-λCGD: Efficient Noise Correlation for Differentially Private Model Training](https://arxiv.org/abs/2601.22334) | — |
| [PINA](math/pina.md) | ICASSP 2026 | [Differentially Private Clustered Federated Learning with Privacy-Preserving Initialization and Normality-Driven Aggregation](https://arxiv.org/abs/2604.20596) | — |
| [RaCO-DP](math/racodp.md) | ICLR 2026 | [Private Rate-Constrained Optimization with Applications to Fair Learning](https://arxiv.org/abs/2505.22703) | [official](https://github.com/cleverhans-lab/dp-raco) |
| [DP-MacAdam](math/dpmacadam.md) | arXiv 2026 | [DP-MacAdam: Differentially Private Mechanism with Adaptive Clipping and Adaptive Momentum](https://arxiv.org/abs/2606.05435) | — | — |
| [FO-DP-SGD](math/fodpsgd.md) | arXiv 2026 | [Deep Learning under Fractional-Order Differential Privacy](https://arxiv.org/abs/2605.09890) | — | — |
| [Hyperparameter-free DP optimization (GeN-DP)](math/hyperparameterfreedpoptimizationgendp.md) | ICLR 2025 | [Towards hyperparameter-free optimization with differential privacy](https://arxiv.org/abs/2503.00703) | — | — |
| [DP-Muon](math/dpmuon.md) | arXiv 2026 | [DP-Muon: Differentially Private Optimization via Matrix-Orthogonalized Momentum](https://arxiv.org/abs/2605.12994) | — | — |
| [TP-TopK](math/tptopk.md) | arXiv 2026 | [When Do Fewer Coordinates Suffice in DP-SGD?](https://arxiv.org/abs/2606.04375) | — | — |
| [DPDL](math/dpdl.md) | arXiv 2026 | [DPDL: Towards Differential Privacy Preservation in Decentralized Stochastic Learning on Non-IID Data](https://arxiv.org/abs/2606.04399) | — | — |
| [DP-SGD-RC](math/dpsgdrc.md) | ICML 2026 | [Efficient DP-SGD for LLMs with Randomized Clipping](https://arxiv.org/abs/2605.24879) | — | — |
| [PRISM](math/prism2.md) | ICML 2026 | [PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](https://arxiv.org/abs/2606.00944) | — | — |
| [SMA-DP-SGD](math/smadpsgd.md) | arXiv 2026 | [SMA-DP: Spectral Memory-Aware Differential Privacy for Deep Learning](https://arxiv.org/abs/2605.20450) | — | — |
| [FiBeR](math/fiber.md) | arXiv 2026 | [FIBER: A Differentially Private Optimizer with Filter-Aware Innovation Bias Correction](https://arxiv.org/abs/2605.03425) | — | — |
| [DP-KFC](math/dpkfc.md) | ICML 2026 | [DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](https://arxiv.org/abs/2605.13418) | [official](https://github.com/molinamarcvdb/DP-KFC) | — |
| [DP-FedAdamW](math/dpfedadamw.md) | CVPR 2026 | [DP-FedAdamW: An Efficient Optimizer for Differentially Private Federated Large Models](https://arxiv.org/abs/2602.19945) | — | — |
| [Lap2](math/lap2.md) | IEEE CSF 2026 | [Lap2: Revisiting Laplace DP-SGD for High Dimensions via Majorization Theory](https://arxiv.org/abs/2602.23516) | [official](https://github.com/datasec-lab/lap2) | — |
| [Clip21-SGD2M](math/clip21sgd2m.md) | arXiv 2025 | [Double Momentum and Error Feedback for Clipping with Fast Rates and Differential Privacy](https://arxiv.org/abs/2502.11682) | — | — |
