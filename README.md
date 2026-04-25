QMDD: Adversarial Robustness of Quantized Deepfake Detection
This README provides an overview of the QMDD project pipeline, based on the IJCAI 2025 paper, "Unlocking the Potential of Lightweight Quantized Models for Deepfake Detection". This implementation evaluates the adversarial robustness of lightweight, quantized models trained for binary classification.

🚀 Overview
The provided pipeline trains a lightweight deepfake detection model and evaluates its resilience against adversarial attacks. It explicitly investigates the trade-off between model quantization and adversarial vulnerability by analyzing the Shifted Logarithmic Redistribution (SLR) quantizer.

Key Specifications
Reference Paper: Unlocking the Potential of Lightweight Quantized Models for Deepfake Detection (IJCAI 2025).


Runtime Requirement: Requires a GPU environment (T4 GPU recommended).

🛠️ Model Architecture & Methodology
The project relies on a globally defined QMDDModel class that mimics the QMDD ResNet-23 CQB structure.

Backbone: Utilizes a truncated ResNet-18 architecture, specifically extracting features from the stem and layers 1 through 3.

Quantization: Implements a custom 2-bit Shifted Logarithmic Redistribution (SLR) quantizer after each residual layer.

The SLR quantizer unfolds near-zero activations via a logarithmic mapping before uniform quantization is applied.

It utilizes a learnable, layer-wise offset parameter called tau.

Classifier: A two-layer Multi-Layer Perceptron (MLP) with a dropout rate of 0.3, reducing the 256-dimensional features to a single binary output.

Training Details: Trained for 10 epochs using the Adam optimizer (learning rate of 1e-4) and BCEWithLogitsLoss.

🛡️ Adversarial Evaluation Suite
The model is evaluated against standard adversarial attacks to measure its robustness.

Attacks Implemented: Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

Epsilon Values: The robustness is tested across epsilons of 4/255, 8/255, and 16/255.

Adversarial-SLR tau Sweep: A core research contribution involves sweeping the tau parameter across values of 0.0, 0.5, 1.0, 2.0, 4.0, and 8.0.

The sweep tests whether tuning tau improves adversarial robustness.

Higher tau values lead to a wider logarithmic redistribution and smoother decision boundaries, which can potentially increase robustness.

The code identifies the "best" tau by minimizing the Robustness Gap (the difference between Clean Accuracy and PGD Accuracy).

💾 Saving & Outputs
To ensure compatibility with PyTorch 2.6+ and to avoid UnpicklingError issues, the script safely saves and loads model weights using model.state_dict() and weights_only=True.

Running the pipeline will generate a zipped archive (qmdd_results.zip) containing the following artifacts:

Model Weights: qmdd_cifar10_state.pth.

Data Logs: adversarial_robustness.csv and tau_sweep.csv detailing the exact accuracy drops across experiments.

Visualizations: * dataset_samples.png (Real/Fake CIFAR-10 breakdown).

training_curves.png (BCE Loss and Accuracy history).

confusion_matrix.png (Performance on the clean test set).

full_results.png (A comprehensive 2x2 grid plotting the tau sweep, robustness gaps, epsilon attack comparisons, and training history).
