# Fractional-Order Optimizers

Fractional-order optimizers generalize the integer-order gradient step with fractional-calculus operators, most commonly the Caputo, Riemann-Liouville, or Grünwald-Letnikov derivative, which weight past gradient information through power-law memory kernels. The field is young: the first neural-network training results date to 2015, convergence theory is still being settled, and most papers ship no code.

## Foundations

| Optimizer | Venue | Paper | Code |
|---|---|---|---|
| [the Fractional Steepest Descent Method (FSDM)](math/x243.md) | IEEE Transactions on Neural Networks and Learning Systems 2015 | [Fractional Extreme Value Adaptive Training Method: Fractional Steepest Descent Approach](https://doi.org/10.1109/TNNLS.2013.2286175) | — |
| [Caputo BP-NN FOGD (ISNN)](math/caputobpnnfogdisnn.md) | Advances in Neural Networks - ISNN 2017 (Lecture Notes in Computer Science) | [A Caputo-Type Fractional-Order Gradient Descent Learning of BP Neural Networks](https://doi.org/10.1007/978-3-319-59072-1_64) | — |
| [Caputo CVNN FOGD](math/caputocvnnfogd.md) | IEEE Access 2017 | [Convergence Analysis of Caputo-Type Fractional Order Complex-Valued Neural Networks](https://doi.org/10.1109/ACCESS.2017.2679185) | — |
| [Caputo fractional-order gradient descent](math/x246.md) | Neural Networks 2017 | [Fractional-order gradient descent learning of BP neural networks with Caputo derivative](https://doi.org/10.1016/j.neunet.2017.02.007) | — |
| [FBPTT](math/fbptt.md) | Circuits, Systems, and Signal Processing 2018 | [A Novel Fractional Gradient-Based Learning Algorithm for Recurrent Neural Networks](https://doi.org/10.1007/s00034-017-0572-z) | — |
| [FGD-RBF](math/fgdrbf.md) | Circuits, Systems, and Signal Processing 2018 | [A Fractional Gradient Descent-Based RBF Neural Network](https://doi.org/10.1007/s00034-018-0835-3) | — |
| [Fractional-Order Deep BP NN](math/fractionalorderdeepbpnn.md) | Computational Intelligence and Neuroscience 2018 | [Fractional-Order Deep Backpropagation Neural Network](https://doi.org/10.1155/2018/7361628) | [official](https://github.com/BaoChunhui/Deep-fractional-BP-neural-networks) |
| [Caputo-Type FOGD (Deep BP)](math/caputotypefogddeepbp.md) | IEEE IMCEC 2019 | [A Caputo-Type Fractional-Order Gradient Descent Learning of Deep BP Neural Networks](https://doi.org/10.1109/IMCEC46724.2019.8984089) | — |
| [FSGD](math/fsgd.md) | Electronic Markets 2019 | [Fractional stochastic gradient descent for recommender systems](https://doi.org/10.1007/s12525-018-0297-2) | — |
| [mF-SGD](math/mfsgd.md) | IEEE Access 2019 | [Design of Momentum Fractional Stochastic Gradient Descent for Recommender Systems](https://doi.org/10.1109/ACCESS.2019.2954859) | — |
| [CFEM-LMS](math/cfemlms.md) | Neurocomputing 2020 | [Combination of fractional FLANN filters for solving the Van der Pol-Duffing oscillator](https://doi.org/10.1016/j.neucom.2020.02.022) | — |
| [FSDM](math/fsdm.md) | Frontiers of Information Technology & Electronic Engineering 2020 | [Fractional-order global optimal backpropagation machine trained by an improved fractional-order steepest descent method](https://doi.org/10.1631/FITEE.1900593) | — |
| [Fractional Order Gradient Method](math/x255.md) | Neurocomputing 2020 | [Convolutional neural networks with fractional order gradient method](https://arxiv.org/abs/1905.05336) | — |
| [Normalized Fractional SGD (NFSGD)](math/x256.md) | Neural Computing and Applications 2020 | [Design of normalized fractional SGD computing paradigm for recommender systems](https://doi.org/10.1007/s00521-019-04562-6) | — |
| [the Fractional Order Gradient Method](math/x257.md) | Journal of the Franklin Institute 2020 | [Generalization of the gradient method with fractional order gradient direction](https://doi.org/10.1016/j.jfranklin.2020.01.008) | — |
| [Fractional Order Gradient Descent with Momentum (FOGDM)](math/x258.md) | Network: Computation in Neural Systems 2020 | [Data classification based on fractional order gradient descent with momentum for RBF neural network](https://doi.org/10.1080/0954898X.2020.1849842) | — |
| [CFGD (Caputo)](math/cfgdcaputo.md) | arXiv 2021 | [A Caputo fractional derivative-based algorithm for optimization](https://arxiv.org/abs/2104.02259) | — |
| [Fractional-Order Momentum (FCM)](math/x260.md) | Neurocomputing 2021 | [Convolutional neural networks based on fractional-order momentum for parameter training](https://doi.org/10.1016/j.neucom.2021.03.075) | — |
| [FOGDM-RBF](math/x261.md) | Soft Computing 2021 | [Fractional-order gradient descent with momentum for RBF neural network-based AIS trajectory restoration](https://doi.org/10.1007/s00500-020-05484-5) | — |
| [Caputron](math/caputron.md) | Electronics (MDPI) 2022 | [Exploring the Effects of Caputo Fractional Derivative in Spiking Neural Network Training](https://doi.org/10.3390/electronics11142114) | [official](https://github.com/nata108/Caputron) |
| [FGD (CNN BP)](math/fgdcnnbp.md) | arXiv 2022 | [Using a novel fractional-order gradient method for CNN back-propagation](https://arxiv.org/abs/2205.00581) | — |
| [FGNN](math/fgnn.md) | Mathematics (MDPI) 2022 | [A Regularized Graph Neural Network Based on Approximate Fractional Order Gradients](https://doi.org/10.3390/math10081320) | — |
| [FracM](math/fracm.md) | Neural Computing and Applications 2022 | [A fractional-order momentum optimization approach of deep neural networks](https://doi.org/10.1007/s00521-021-06765-2) | [community](https://github.com/TruongChien/FracM) |
| [GFSGD](math/gfsgd.md) | Chaos, Solitons & Fractals 2022 | [Generalized fractional strategy for recommender systems with chaotic ratings behavior](https://doi.org/10.1016/j.chaos.2022.112204) | — |
| [Fractional Derivative Gradient Optimizers (FSGD](math/x267.md) | Applied Sciences 2022 | [Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition](https://doi.org/10.3390/app12189264) | — |
| [Fractional LMS (FLMS)](math/x268.md) | IEEE Transactions on Signal Processing 2022 | [Performance Analysis of Fractional Learning Algorithms](https://arxiv.org/abs/2110.05201) | — |
| [Conformable Fractional Gradient Descent](math/x269.md) | Fuzzy Systems and Data Mining VIII 2022 | [Fractional Gradient Descent Learning of Backpropagation Artificial Neural Networks with Conformable Fractional Calculus](https://doi.org/10.3233/FAIA220372) | — |
| [Fractional Order Gradient Descent with variable initial value](math/x270.md) | Neurocomputing 2022 | [Study on fast speed fractional order gradient descent method and its application in neural networks](https://doi.org/10.1016/j.neucom.2022.02.034) | — |
| [TFGD (Time-fractional)](math/tfgdtimefractional.md) | Axioms 2022 | [Training Neural Networks by Time-Fractional Gradient Descent](https://doi.org/10.3390/axioms11100507) | — |
| [Variable Order Fractional Gradient Descent](math/x272.md) | Chinese Control and Decision Conference 2022 | [Variable Order Fractional Gradient Descent Method and Its Application in Neural Networks Optimization](https://doi.org/10.1109/CCDC55256.2022.10033456) | — |
| [CfGD / CfAdam](math/cfgdcfadam.md) | Neural Networks 2023 | [Accelerating gradient descent and Adam via fractional gradients](https://doi.org/10.1016/j.neunet.2023.01.002) | — |
| [RFGD](math/rfgd.md) | Neural Networks 2023 | [A fractional gradient descent algorithm robust to the initial weights of multilayer perceptron](https://doi.org/10.1016/j.neunet.2022.11.018) | — |
| [FO-RI-FedAvg](math/forifedavg.md) | arXiv 2026 | [Fractional Order Federated Learning for Battery Electric Vehicle Energy Consumption Modeling](https://arxiv.org/abs/2602.12567) | — | — |
| [IHL-Adam](math/ihladam.md) | Expert Systems with Applications 2024 | [Parameter training method for convolutional neural networks based on improved Hausdorff-like derivative](https://doi.org/10.1016/j.eswa.2023.121659) | — | — |

## Recent advances

| Optimizer | Venue | Paper | Code |
|---|---|---|---|
| [AFOGD / AFOAGD](math/afogdafoagd.md) | arXiv 2023 | [The Novel Adaptive Fractional Order Gradient Decent Algorithms Design via Robust Control](https://arxiv.org/abs/2303.04328) | — |
| [EFSGD / EN-EFSGD](math/efsgdenefsgd.md) | Chaos, Solitons & Fractals 2023 | [Enhanced fractional prediction scheme for effective matrix factorization in chaotic feedback recommender systems](https://doi.org/10.1016/j.chaos.2023.114109) | — |
| [FCGD_G-L](math/fcgdgl.md) | Mathematics 2023 | [A Deep Learning Optimizer Based on Grünwald–Letnikov Fractional Order Definition](https://doi.org/10.3390/math11020316) | — |
| [FGDAM](math/fgdam.md) | Applied Mathematics and Computation 2023 | [Applications of fractional gradient descent method with adaptive momentum in BP neural networks](https://doi.org/10.1016/j.amc.2023.127944) | — |
| [FracG](math/fracg.md) | Chinese Control Conference (CCC) 2023 | [Optimization Method of Neural Networks via Fractional-Order of Gradients](https://doi.org/10.23919/CCC58697.2023.10239893) | — |
| [Fractional Gradient Descent (FSGD)](math/x282.md) | Fractal and Fractional 2023 | [Fractional Gradient Optimizers for PyTorch: Enhancing GAN and BERT](https://doi.org/10.3390/fractalfract7070500) | — |
| [the Improved Stochastic Fractional Order Gradient Descent algorithm](math/x283.md) | Fractal and Fractional 2023 | [The Improved Stochastic Fractional Order Gradient Descent Algorithm](https://doi.org/10.3390/fractalfract7080631) | — |
| [AdaGL](math/adagl.md) | Neural Processing Letters 2024 | [An Adaptive Learning Rate Deep Learning Optimizer Using Long and Short-Term Gradients Based on G–L Fractional-Order Derivative](https://doi.org/10.1007/s11063-024-11571-7) | [community](https://github.com/daddydrac/AdaGL) |
| [GFSGD](math/gfsgd2.md) | Heliyon 2024 | [Fractional gradient optimized explainable convolutional neural network for Alzheimer's disease diagnosis](https://doi.org/10.1016/j.heliyon.2024.e39037) | — |
| [FOAdam](math/x286.md) | Applied Mathematical Modelling 2024 | [A novel gradient descent optimizer based on fractional order scheduler and its application in deep neural networks](https://doi.org/10.1016/j.apm.2023.12.018) | — |
| [Adaptive Terminal Caputo Fractional Gradient Descent (AT-CFGD)](math/x287.md) | TMLR 2024 | [Convergence Analysis of Fractional Gradient Descent](https://arxiv.org/abs/2311.18426) | — |
| [Caputo Fractional-Order Gradient Descent](math/x288.md) | International Journal of Fuzzy Systems 2024 | [A Novel Neuro-fuzzy Learning Algorithm for First-Order Takagi–Sugeno Fuzzy Model: Caputo Fractional-Order Gradient Descent Method](https://doi.org/10.1007/s40815-024-01750-y) | — |
| [FNGD](math/fngd.md) | IEEE Access 2024 | [Improving the Accuracy of Neural Network Pattern Recognition by Fractional Gradient Descent](https://doi.org/10.1109/ACCESS.2024.3491614) | — |
| [MFFGD](math/mffgd.md) | Neurocomputing 2024 | [MFFGD: An adaptive Caputo fractional-order gradient algorithm for DNN](https://doi.org/10.1016/j.neucom.2024.128606) | — |
| [Caputo-based SGD (L1 scheme)](math/caputobasedsgdl1scheme.md) | OpenReview 2024 | [Stochastic Fractional Gradient Descent with Caputo L1 Scheme for Deep Neural Networks](https://openreview.net/forum?id=hCGaySEW9q) | — |
| [C-FOG](math/cfog.md) | Fractal and Fractional 2024 | [Self-Organizing Optimization Based on Caputo's Fractional Order Gradients](https://doi.org/10.3390/fractalfract8080451) | — |
| [CSA-CFGD](math/csacfgd.md) | PeerJ Computer Science 2024 | [Deep ocular tumor classification model using cuckoo search algorithm and Caputo fractional gradient descent](https://doi.org/10.7717/peerj-cs.1923) | [official](https://doi.org/10.7717/peerj-cs.1923/supp-1) |
| [FGD-RBFNN (UAV)](math/fgdrbfnnuav.md) | Computer Modeling in Engineering & Sciences 2024 | [Fractional Gradient Descent RBFNN for Active Fault-Tolerant Control of Plant Protection UAVs](https://doi.org/10.32604/cmes.2023.030535) | — |
| [FOELM](math/foelm.md) | Applied Soft Computing 2024 | [An interval neural network-based Caputo fractional-order extreme learning machine applied to classification](https://doi.org/10.1016/j.asoc.2024.112310) | — |
| [MIF](math/mif.md) | Algorithms 2024 | [An Integer-Fractional Gradient Algorithm for Back Propagation Neural Networks](https://doi.org/10.3390/a17050220) | — |
| [Multi-layer NN FOGD](math/multilayernnfogd.md) | Advanced Theory and Simulations 2024 | [Convergence Analysis and Application for Multi-Layer Neural Network Based on Fractional-Order Gradient Descent Learning](https://doi.org/10.1002/adts.202300662) | — |
| [UCAdam](math/ucadam.md) | Journal of Electrical Systems 2024 | [Improved Adam: Incorporating Unified Conformable Fractional Derivative for fractional-order Momentum](https://journal.esrgroups.org/jes/article/view/5687) | — |
| [2SEDFOSGD](math/2sedfosgd.md) | arXiv 2025 | [Effective Dimension Aware Fractional-Order Stochastic Gradient Descent for Convex Optimization Problems](https://arxiv.org/abs/2503.13764) | — |
| [2SEDFOSGD](math/2sedfosgd2.md) | arXiv 2025 | [More Optimal Fractional-Order Stochastic Gradient Descent for Non-Convex Optimization Problems](https://arxiv.org/abs/2505.02985) | — |
| [AFGD (adaptive Caputo FGD for TCN)](math/afgdadaptivecaputofgdfortcn.md) | Neurocomputing 2025 | [Monotonic convergence of adaptive Caputo fractional gradient descent for temporal convolutional networks](https://doi.org/10.1016/j.neucom.2025.131491) | — |
| [FGDSINN](math/fgdsinn.md) | International Journal of Machine Learning and Cybernetics 2025 | [A smoothing interval neural networks-based Caputo fractional-order gradient learning algorithm](https://doi.org/10.1007/s13042-024-02402-1) | — |
| [FOSGD / FOSGDM / FOSGDME](math/fosgdfosgdmfosgdme.md) | Neural Networks 2025 | [Fractional-order stochastic gradient descent method with momentum and energy for deep neural networks](https://doi.org/10.1016/j.neunet.2024.106810) | — |
| [FracGrad](math/fracgrad.md) | Fractal and Fractional 2025 | [FracGrad: A Discretized Riemann–Liouville Fractional Integral Approach to Gradient Accumulation for Deep Learning](https://doi.org/10.3390/fractalfract9110733) | — |
| [GF-SGD](math/gfsgd3.md) | Computers in Biology and Medicine 2025 | [Generalized fractional optimization-based explainable lightweight CNN model for malaria disease classification](https://doi.org/10.1016/j.compbiomed.2024.109593) | — |
| [IFOGD](math/ifogd.md) | Neural Networks 2025 | [Improved fractional-order gradient descent method based on multilayer perceptron](https://doi.org/10.1016/j.neunet.2024.106970) | — |
| [L2O-CFGD](math/l2ocfgd.md) | arXiv 2025 | [Enhancing Fractional Gradient Descent with Learned Optimizers](https://arxiv.org/abs/2510.18783) | [official](https://github.com/Johnny1188/fractional-learning-to-optimize) |
| [MOAOCFGD](math/moaocfgd.md) | arXiv 2025 | [An Adaptive Order Caputo Fractional Gradient Descent Method for Multi-objective Optimization Problems](https://arxiv.org/abs/2507.07674) | — |
| [NCFDD / NFLightGBM](math/ncfddnflightgbm.md) | Information Fusion 2025 | [Fractional light gradient boosting machine ensemble learning model: A non-causal fractional difference descent approach](https://doi.org/10.1016/j.inffus.2025.102947) | — |
| [a Caputo fractional-order gradient descent for neural network training](math/x310.md) | Chaos, Solitons & Fractals 2025 | [Fractional-order gradient approach for optimizing neural networks: A theoretical and empirical analysis](https://doi.org/10.1016/j.chaos.2025.116009) | — |
| [Fractional-order SGD (FSGD)](math/x311.md) | arXiv 2025 | [Fractional-order Jacobian Matrix Differentiation and Its Application in Artificial Neural Networks](https://arxiv.org/abs/2506.07408) | — |
| [Adaptive Parameter Fractional-Order Gradient Descent Learning](math/x312.md) | European Journal of Operational Research 2025 | [Novel adaptive parameter fractional-order gradient descent learning for stock selection decision support systems](https://doi.org/10.1016/j.ejor.2025.01.013) | — |
| [FAdam](math/x313.md) | Chaos, Solitons & Fractals 2025 | [Parameter training methods for convolutional neural networks with adaptive adjustment method based on Caputo fractional-order differences](https://doi.org/10.1016/j.chaos.2025.116588) | — |
| [SFM](math/sfm.md) | Digital Signal Processing 2025 | [A momentum-based stochastic fractional gradient optimizer with U-net model for brain tumor segmentation in MRI](https://doi.org/10.1016/j.dsp.2025.104983) | — |
| [Caputo Fractional-order Gradient Descent for Ridge Polynomial Neural](math/x315.md) | International Conference on Electronics and Communication, Network and Computer Technology 2025 | [A Novel Method for Ridge Polynomial Neural Network-based Caputo Fractional-order Gradient Descent Algorithm](https://doi.org/10.1109/ECNCT66493.2025.11172593) | — |
| [AOFGD](math/aofgd.md) | SSRN 2025 | [AOFGD: Adaptive order fractional gradient descent method](https://doi.org/10.2139/ssrn.5717167) | — |
| [Frac-Adam](math/x317.md) | Mathematics 2025 | [Fractional Optimizers for LSTM Networks in Financial Time Series Forecasting](https://doi.org/10.3390/math13132068) | — |
| [Caputo Fractional Gradient Descent](math/x318.md) | International Conference on Advanced Algorithms and Control Engineering 2025 | [Fractional Order Gradient Descent with Caputo Derivatives for Product-Unit Neural Networks](https://doi.org/10.1109/ICAACE65325.2025.11020545) | — |
| [FO-STDGD](math/fostdgd.md) | Neurocomputing 2025 | [Fractional-order spike-timing-dependent gradient descent for multi-layer spiking neural networks](https://doi.org/10.1016/j.neucom.2024.128662) | — |
| [Fractional Order Stochastic Gradient Descent (FOSGD)](math/x320.md) | ASME IDETC-CIE 2025 | [Tail-Index-Awareness in Fractional Order Stochastic Gradient Descent](https://doi.org/10.1115/DETC2025-169054) | — |
| [λ-FAdaMax](math/fadamax.md) | Expert Systems with Applications 2025 | [λ-FAdaMax: A novel fractional-order gradient descent method with decaying second moment for neural network training](https://doi.org/10.1016/j.eswa.2025.127156) | — |
| [CFDNN](math/cfdnn.md) | Scientific Reports 2026 | [Conformable Fractional Deep Neural Networks (CFDNN) for high-speed cyber-attack detection](https://doi.org/10.1038/s41598-026-45213-w) | — |
| [CFGD (Compressed)](math/cfgdcompressed.md) | IEEE Transactions on Neural Networks and Learning Systems 2026 | [Fractional Gradient Descent With Matrix Stepsizes for Non-Convex Optimization](https://doi.org/10.1109/TNNLS.2025.3637535) | [official](https://github.com/alokendumazumder/IEEE_TNNLS_Compressed_fractional_GD) |
| [FAdamWav](math/fadamwav.md) | Fractal and Fractional 2026 | [FAdamWav: A Fractional Wavelet Gradient Optimizer for Neural Networks](https://doi.org/10.3390/fractalfract10030149) | — |
| [FOFedAvg](math/fofedavg.md) | arXiv 2026 | [Fractional-Order Federated Learning](https://arxiv.org/abs/2602.15380) | — |
| [Fractional-order FL with adaptive momentum](math/fractionalorderflwithadaptivemomentum.md) | IEEE Transactions on Emerging Topics in Computational Intelligence 2026 | [Communication-Efficient Federated Learning via Fractional-Order Gradient Descent With Adaptive Momentum Under Non-IID Data](https://doi.org/10.1109/TETCI.2026.3692489) | — |
| [TFGD (Tempered)](math/tfgdtempered.md) | Neural Networks 2026 | [Tempered fractional gradient descent: Theory, algorithms, and robust learning applications](https://doi.org/10.1016/j.neunet.2025.108005) | — |
| [FGD-ED](math/x328.md) | Information Processing & Management 2026 | [Fractional-order gradient descent method based on fractional-order term exponential decay and its application in artificial neural networks](https://doi.org/10.1016/j.ipm.2025.104448) | — |
| [the Caputo Fractional-Order Gradient Descent Method (FGDM)](math/x329.md) | Applied Soft Computing 2026 | [A novel gradient learning algorithm based on zero-order Takagi-Sugeno fuzzy model: the caputo fractional-order gradient descent](https://doi.org/10.1016/j.asoc.2025.114430) | — |
| [CFGD (Conformable)](math/cfgdconformable.md) | Journal of Computational and Applied Mathematics 2026 | [Conformable fractional gradient descent: A local optimizer for neural network training](https://doi.org/10.1016/j.cam.2026.117842) | — |
| [NGLFGD](math/x331.md) | Knowledge-Based Systems 2026 | [Fast and accurate fractional order gradient descent algorithm and its application in Extreme Gradient Boosting](https://doi.org/10.1016/j.knosys.2025.114911) | — |
| [FO-Elman](math/x332.md) | Neural Networks 2026 | [Fractional-order gradient descent learning for Elman neural networks](https://doi.org/10.1016/j.neunet.2026.108880) | — |

## Surveys

| Optimizer | Venue | Paper | Code |
|---|---|---|---|
| [Fractional-Order Gradient Descent for Neural Networks](math/x333.md) | The European Physical Journal Special Topics 2022 | [Artificial neural networks: a practical review of applications involving fractional calculus](https://doi.org/10.1140/epjs/s11734-022-00455-3) | — |
| [Fractional Gradient Descent (FGD)](math/x334.md) | Chaos, Solitons & Fractals 2025 | [A comprehensive survey of fractional gradient descent methods and their convergence analysis](https://doi.org/10.1016/j.chaos.2025.116154) | — |
| [the Fractional Continuous Time Method (FCTM)](math/x335.md) | Journal of Computational and Applied Mathematics 2026 | [An overview of the fractional-order gradient descent method and its applications](https://arxiv.org/abs/2601.03318) | — |

Note: FAdam ([arXiv 2405.12807](https://arxiv.org/abs/2405.12807)) is a Fisher-information variant of Adam and is unrelated to fractional calculus despite the name.
