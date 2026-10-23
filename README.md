# 🌸 Multi-Layer Perceptron (MLP) from Scratch on Iris Dataset

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Pure%20Math-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Validation-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow%2FKeras-Benchmark-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.67%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, mathematically rigorous implementation of a **Multi-Layer Perceptron (MLP) Neural Network built entirely from scratch using Pure NumPy**—no black-box deep learning frameworks used for forward propagation, loss calculation, or backpropagation. 

The scratch model is thoroughly validated, diagnosed with rich visualizations, and benchmarked against **TensorFlow/Keras** and **Scikit-Learn** (`MLPClassifier`), achieving a matching **96.67% Test Accuracy**.

---

## 📑 Table of Contents
- [Architecture & Design](#-architecture--design)
- [Mathematical Formulation](#-mathematical-formulation)
- [Summary: How We Optimized Accuracy (From 63% to 96.7%)](#-summary-how-we-optimized-accuracy-from-63-to-967)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Performance & Evaluation](#-performance--evaluation)
- [Benchmark: Scratch MLP vs. Keras & Scikit-Learn](#-benchmark-scratch-mlp-vs-keras--scikit-learn)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [LinkedIn Post Template](#-ready-to-share-linkedin-post-template)

---

## 🧠 Architecture & Design

The network is structured as a **3-layer fully connected Feedforward Neural Network**:
- **Input Layer**: $4$ nodes corresponding to continuous flower measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
- **Hidden Layer**: $8$ neurons with non-linear **ReLU** activation.
- **Output Layer**: $3$ neurons with **Softmax** activation representing the probability distribution across the 3 Iris species (*Setosa*, *Versicolor*, *Virginica*).

![MLP Architecture](assets/mlp_architecture.png)

### Parameter Matrix Dimensions
| Layer Connection | Weight Matrix $W$ | Bias Vector $b$ | Activation Function |
|:---|:---:|:---:|:---:|
| **Input $\rightarrow$ Hidden** | $W^{[1]} \in \mathbb{R}^{4 \times 8}$ | $b^{[1]} \in \mathbb{R}^{1 \times 8}$ | $\mathrm{ReLU}(z) = \max(0, z)$ |
| **Hidden $\rightarrow$ Output** | $W^{[2]} \in \mathbb{R}^{8 \times 3}$ | $b^{[2]} \in \mathbb{R}^{1 \times 3}$ | $\mathrm{Softmax}(z)$ |

---

## 📐 Mathematical Formulation

### 1. Forward Propagation
Given a batch of $m$ input samples $X \in \mathbb{R}^{m \times 4}$:
1. **Hidden Layer Pre-activation**:
   $$Z^{[1]} = X W^{[1]} + b^{[1]}$$
2. **Hidden Layer Activation**:
   $$A^{[1]} = \mathrm{ReLU}(Z^{[1]}) = \max(0, Z^{[1]})$$
3. **Output Layer Pre-activation**:
   $$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}$$
4. **Output Probabilities (Softmax)**:
   $$\hat{Y}_{ij} = \frac{e^{Z^{[2]}_{ij} - \max_k(Z^{[2]}_{ik})}}{\sum_{k=1}^3 e^{Z^{[2]}_{ik} - \max_k(Z^{[2]}_{ik})}}$$
   *(Shift invariance trick: subtracting row maximums prevents numerical floating-point overflow).*

### 2. Categorical Cross-Entropy Loss
For one-hot encoded ground truth $Y \in \{0, 1\}^{m \times 3}$:
$$\mathcal{L}(Y, \hat{Y}) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^3 Y_{ik} \log(\hat{Y}_{ik} + \epsilon)$$
*(where $\epsilon = 10^{-15}$ guards against $\log(0)$).*

### 3. Backpropagation via Analytical Gradients
1. **Output Layer Error ($\delta^{[2]}$)**:
   $$\delta^{[2]} = \frac{\partial \mathcal{L}}{\partial Z^{[2]}} = \hat{Y} - Y \quad (m \times 3)$$
2. **Output Layer Parameter Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m} (A^{[1]})^T \delta^{[2]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m} \sum_{i=1}^m \delta^{[2]}_{i,:}$$
3. **Hidden Layer Error ($\delta^{[1]}$)**:
   $$\delta^{[1]} = (\delta^{[2]} (W^{[2]})^T) \odot \mathrm{ReLU}'(Z^{[1]}) \quad \text{where } \mathrm{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$$
4. **Hidden Layer Parameter Gradients**:
   $$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m} X^T \delta^{[1]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m} \sum_{i=1}^m \delta^{[1]}_{i,:}$$
5. **Batch Gradient Descent Updates**:
   $$W^{[l]} := W^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[l]}}, \qquad b^{[l]} := b^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[l]}}$$

---

## 💡 Summary: How We Optimized Accuracy (From 63% to 96.7%)

In naive or tutorial implementations, MLP models built from scratch often stall around **60%–67% accuracy** (failing to separate *Versicolor* and *Virginica*). Below is a comprehensive breakdown of the architectural and numerical enhancements introduced to achieve **96.67% Test Accuracy** and **97.50% Training Accuracy**:

| Optimization Component | Naive Baseline (Before) | Engineered Model (Current) | Why It Matters / Technical Impact |
|:---|:---|:---|:---|
| **1. Feature Scaling** | Unscaled raw values (`4.3` to `7.9` cm vs `0.1` to `2.5` cm) | **`StandardScaler`** ($z = \frac{x - \mu}{\sigma}$) | Eliminates elongated loss contours and gradient oscillations. Balances weight updates across all 4 feature dimensions. |
| **2. Weight Initialization** | Tiny Gaussian noise ($W \times 0.01$) | **He (Kaiming) & Xavier Normal** | Prevents vanishing activations in deep ReLU networks. Calibrates layer variance: $\mathrm{Var}(W) = \frac{2}{n_{in}}$. |
| **3. Numerical Stability** | Raw `np.exp()` and unclipped `np.log()` | **Shift-invariant Softmax & Log Clipping** | Prevents `inf` during exponential sums and avoids `NaN` errors when predicted probabilities approach zero. |
| **4. Learning Rate & Epochs** | $\alpha = 0.01$ or $0.04$ | **Tuned $\alpha = 0.08$ (400 Epochs)** | Provides smooth, monotonically decreasing loss curves without overshoot or premature plateauing. |
| **5. Data Splitting** | Unstratified random split | **Stratified 80/20 Train-Test Split** | Ensures identical class proportions (10 samples/class in test) for unbiased evaluation. |

---

## 📊 Exploratory Data Analysis (EDA)

Before feeding the dataset into our neural network, we perform exploratory data analysis to inspect distributions, correlations, and class separability:

![EDA Dashboard](assets/eda_dashboard.png)

![Pairplot](assets/eda_pairplot.png)

- **Iris-Setosa** is linearly separable using Petal dimensions alone.
- **Iris-Versicolor** and **Iris-Virginica** exhibit non-linear boundary overlap, which the hidden layer non-linearity (ReLU) successfully maps and resolves.

---

## 🚀 Performance & Evaluation

### Training & Validation Loss/Accuracy Curves
The model converges cleanly without signs of overfitting or instability across 400 epochs of batch gradient descent:

![Training Curves](assets/training_validation_curves.png)

### Test Set Metrics (Held-out 30 samples)
```text
=============================================
Final Training Loss     : 0.0858
Final Training Accuracy : 97.50%
Final Test Loss         : 0.1020
Final Test Accuracy     : 96.67%
=============================================
```

### Confusion Matrix & Classification Report
29 out of 30 unseen test instances are classified with 100% precision for *Setosa* and *Versicolor*:

![Confusion Matrix](assets/confusion_matrix.png)

```text
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        10
  versicolor     1.0000    0.9000    0.9474        10
   virginica     0.9091    1.0000    0.9524        10

    accuracy                         0.9667        30
   macro avg     0.9697    0.9667    0.9666        30
weighted avg     0.9697    0.9667    0.9666        30
```

### 2D Decision Boundary (PCA Projection)
Projecting the 4-dimensional feature space onto the top 2 Principal Components showcases the non-linear decision regions mapped by our scratch MLP:

![Decision Boundary](assets/decision_boundary.png)

---

## ⚖️ Benchmark: Scratch MLP vs. Keras & Scikit-Learn

To ensure mathematical parity, we implemented an identical architecture ($4 \rightarrow 8 \rightarrow 3$, ReLU, Softmax, SGD $\alpha = 0.08$) in **TensorFlow/Keras** and **Scikit-Learn (`MLPClassifier`)**:

![Benchmark Comparison](assets/scratch_vs_keras_comparison.png)

| Implementation | Test Accuracy | Architecture | Optimizer | Weight Initialization |
|:---|:---:|:---:|:---:|:---:|
| **Scratch MLP (NumPy)** | **96.67%** | $4 \rightarrow 8 \rightarrow 3$ | Batch Gradient Descent | He / Xavier Normal |
| **Scikit-Learn (`MLPClassifier`)** | **96.67%** | $4 \rightarrow 8 \rightarrow 3$ | Batch Gradient Descent | Glorot Normal |
| **TensorFlow / Keras (SGD)** | **96.67%** | $4 \rightarrow 8 \rightarrow 3$ | Batch Gradient Descent | He / Glorot Normal |

*(Our scratch implementation matches framework-grade accuracy while demonstrating complete transparency over tensor operations).*

---

## 📁 Repository Structure

```text
MLP-IRIS-MODEL/
│
├── assets/                                 # High-resolution architectural diagrams & plots
│   ├── mlp_architecture.png                # Visual diagram of 4 -> 8 -> 3 MLP architecture
│   ├── eda_dashboard.png                   # Feature distributions, boxplots, correlation heatmap
│   ├── eda_pairplot.png                    # Multi-dimensional pairplot of Iris features
│   ├── training_validation_curves.png      # Loss & Accuracy curves over 400 epochs
│   ├── confusion_matrix.png                # Heatmap confusion matrix of test predictions
│   ├── decision_boundary.png               # 2D PCA decision boundary contour plot
│   └── scratch_vs_keras_comparison.png     # Benchmark comparison against Keras & Sklearn
│
├── code_base/                              # Main model implementation
│   └── MLP_Iris_From_Scratch.ipynb         # Cleaned, fully executed, presentation-ready Jupyter Notebook
│
├── data/                                   # Standalone dataset directory
│   └── iris.csv                            # Cleaned Iris dataset (150 rows, 4 features, target, species)
│
├── requirements.txt                        # Pinned dependencies for reproducible execution
└── README.md                               # Project documentation & benchmark summary
```

---

## ⚡ Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ved-hubmax/MLP-IRIS-MODEL.git
cd MLP-IRIS-MODEL
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook code_base/MLP_Iris_From_Scratch.ipynb
```

---

## 💼 Ready-to-Share LinkedIn Post Template

*(Copy and paste this directly to LinkedIn to showcase your project!)*

> 🚀 **Building a Multi-Layer Perceptron (MLP) from Scratch using Pure NumPy (Achieving 96.7% Accuracy!)** 🧠🌸
>
> While modern deep learning frameworks like TensorFlow and PyTorch make building neural networks as easy as stacking API calls, implementing an Artificial Neural Network **from mathematical scratch** offers invaluable intuition into how backpropagation, weight gradients, and loss surfaces behave under the hood.
>
> In this project, I built a 3-layer Multi-Layer Perceptron ($4 \rightarrow 8 \rightarrow 3$) using **only NumPy** to classify the Iris dataset:
>
> 🔹 **Vectorized Forward Propagation**: Handcrafted linear combination ($Z = XW + b$), non-linear ReLU activations, and numerically stable Softmax probability distribution.
> 🔹 **Analytical Backpropagation**: Derived exact partial derivatives via the Chain Rule ($\frac{\partial \mathcal{L}}{\partial W}$, $\frac{\partial \mathcal{L}}{\partial b}$) for both output and hidden layers.
> 🔹 **Accuracy Optimization**: Moving from a naive baseline (~63% accuracy) to a robust **96.67% Test Accuracy** by applying:
>   - Feature Standard Scaling ($z = \frac{x - \mu}{\sigma}$) to prevent gradient oscillations
>   - He (Kaiming) & Xavier weight initialization to eliminate vanishing ReLU gradients
>   - Softmax shift-invariance & probability clipping to guard against numerical overflow
> 🔹 **Benchmarked Against TensorFlow/Keras & Scikit-Learn**: Validated numerical convergence side-by-side with industry frameworks, achieving matching accuracy and loss curves!
>
> 📊 Included full diagnostics: EDA dashboards, pairplots, convergence curves, confusion matrices, and 2D PCA decision boundary visualizations.
>
> 🔗 **GitHub Repository**: https://github.com/ved-hubmax/MLP-IRIS-MODEL
>
> #MachineLearning #DeepLearning #Python #NumPy #DataScience #ArtificialIntelligence #Mathematics #FromScratch #PortfolioProject

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
