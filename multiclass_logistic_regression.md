# Multiclass Logistic Regression — Complete Formula & Calculation Tables

Below is the complete example organized into tables, including every formula, substitution, calculation, and result.

---

## 1. Problem setup

| Parameter | Symbol | Value |
| :--- | :--- | :--- |
| Number of samples | $N$ | 3 |
| Number of features | $D$ | 2 |
| Number of classes | $K$ | 3 |
| Learning rate | $\eta$ | 0.5 |

---

## 2. Input dataset and matrices

| Sample | Feature 1 | Feature 2 | Actual class |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 0 | 1 |
| 2 | 0 | 1 | 2 |
| 3 | 1 | 1 | 3 |

| Matrix | Formula / Values | Shape |
| :--- | :--- | :--- |
| Input $X$ | $\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$ | $3 \times 2$ |
| Actual labels $t$ | $[1, 2, 3]$ | $3$ |
| One-hot $Y$ | $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | $3 \times 3$ |
| Initial weights $W^{(0)}$ | $\begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}$ | $2 \times 3$ |

---

## 3. Forward pass — Logits calculation

**Formula:**

$$Z = XW$$

| Sample | Class 1 calculation | Class 2 calculation | Class 3 calculation | Logits |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $1(1) + 0(0) = 1$ | $1(0) + 0(1) = 0$ | $1(-1) + 0(0) = -1$ | $[1, 0, -1]$ |
| 2 | $0(1) + 1(0) = 0$ | $0(0) + 1(1) = 1$ | $0(-1) + 1(0) = 0$ | $[0, 1, 0]$ |
| 3 | $1(1) + 1(0) = 1$ | $1(0) + 1(1) = 1$ | $1(-1) + 1(0) = -1$ | $[1, 1, -1]$ |

**Final logits matrix:**

$$Z = \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 0 \\ 1 & 1 & -1 \end{bmatrix}$$

---

## 4. Softmax — Convert logits into probabilities

**Formula:**

$$p_{ik} = \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}$$

Exponential values used:

| Logit | Exponential |
| :--- | :--- |
| 1 | 2.7183 |
| 0 | 1.0000 |
| -1 | 0.3679 |

**Detailed calculations:**

| Sample | Class | Exponential numerator | Denominator | Probability |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 2.7183 | 4.0862 | 0.6652 |
| 1 | 2 | 1.0000 | 4.0862 | 0.2447 |
| 1 | 3 | 0.3679 | 4.0862 | 0.0900 |
| 2 | 1 | 1.0000 | 4.7183 | 0.2119 |
| 2 | 2 | 2.7183 | 4.7183 | 0.5761 |
| 2 | 3 | 1.0000 | 4.7183 | 0.2119 |
| 3 | 1 | 2.7183 | 5.8045 | 0.4683 |
| 3 | 2 | 2.7183 | 5.8045 | 0.4683 |
| 3 | 3 | 0.3679 | 5.8045 | 0.0634 |

**Final probability matrix:**

$$P = \begin{bmatrix} 0.6652 & 0.2447 & 0.0900 \\ 0.2119 & 0.5761 & 0.2119 \\ 0.4683 & 0.4683 & 0.0634 \end{bmatrix}$$

---

## 5. Predicted class

**Formula:**

$$\hat{y}_i = \arg\max_k p_{ik}$$

| Sample | Class 1 | Class 2 | Class 3 | Maximum | Predicted class |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.6652 | 0.2447 | 0.0900 | 0.6652 | 1 |
| 2 | 0.2119 | 0.5761 | 0.2119 | 0.5761 | 2 |
| 3 | 0.4683 | 0.4683 | 0.0634 | 0.4683 | 1* |

*\*Sample 3 has a tie between Classes 1 and 2. We select the first maximum.*

$$\hat{y} = [1, 2, 1]$$

---

## 6. Cross-entropy loss calculation

**Individual sample loss formula:**

$$\mathcal{L}_i = -\ln(p_{i,t_i})$$

| Sample | Actual class | Probability selected | Calculation | Loss |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 0.6652 | $-\ln(0.6652)$ | 0.4076 |
| 2 | 2 | 0.5761 | $-\ln(0.5761)$ | 0.5514 |
| 3 | 3 | 0.0634 | $-\ln(0.0634)$ | 2.7586 |

**Average loss formula:**

$$J(W) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i$$

| Calculation | Result |
| :--- | :--- |
| Sum of losses | $0.4076 + 0.5514 + 2.7586$ |
| Total | 3.7176 |
| Number of samples | 3 |
| Average loss | $3.7176 / 3$ |
| **Initial loss** | **1.2392** |

---

## 7. Backward pass — Error calculation

**Formula:**

$$D = P - Y$$

| Sample | Predicted probabilities ($P$) | Actual one-hot ($Y$) | Subtraction | Error |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $[0.6652, 0.2447, 0.0900]$ | $[1, 0, 0]$ | $P - Y$ | $[-0.3348, 0.2447, 0.0900]$ |
| 2 | $[0.2119, 0.5761, 0.2119]$ | $[0, 1, 0]$ | $P - Y$ | $[0.2119, -0.4239, 0.2119]$ |
| 3 | $[0.4683, 0.4683, 0.0634]$ | $[0, 0, 1]$ | $P - Y$ | $[0.4683, 0.4683, -0.9366]$ |

**Final error matrix:**

$$D = \begin{bmatrix} -0.3348 & 0.2447 & 0.0900 \\ 0.2119 & -0.4239 & 0.2119 \\ 0.4683 & 0.4683 & -0.9366 \end{bmatrix}$$

---

## 8. Gradient calculation

**Formula:**

$$\nabla_W J = \frac{1}{N} X^T D$$

### Step 1: Calculate transpose of X

| Original $X$ | Transposed $X^T$ |
| :--- | :--- |
| $\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$ | $\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$ |

### Step 2: Matrix multiplication

$$X^T D = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} -0.3348 & 0.2447 & 0.0900 \\ 0.2119 & -0.4239 & 0.2119 \\ 0.4683 & 0.4683 & -0.9366 \end{bmatrix}$$

| Position | Calculation | Result |
| :--- | :--- | :--- |
| (1,1) | $1(-0.3348) + 0(0.2119) + 1(0.4683)$ | 0.1335 |
| (1,2) | $1(0.2447) + 0(-0.4239) + 1(0.4683)$ | 0.7130 |
| (1,3) | $1(0.0900) + 0(0.2119) + 1(-0.9366)$ | -0.8466 |
| (2,1) | $0(-0.3348) + 1(0.2119) + 1(0.4683)$ | 0.6802 |
| (2,2) | $0(0.2447) + 1(-0.4239) + 1(0.4683)$ | 0.0444 |
| (2,3) | $0(0.0900) + 1(0.2119) + 1(-0.9366)$ | -0.7247 |

$$X^T D \approx \begin{bmatrix} 0.1335 & 0.7130 & -0.8466 \\ 0.6802 & 0.0444 & -0.7247 \end{bmatrix}$$

### Step 3: Divide by N = 3

| Position | Calculation | Gradient |
| :--- | :--- | :--- |
| (1,1) | $0.1335 / 3$ | 0.0445 |
| (1,2) | $0.7130 / 3$ | 0.2377 |
| (1,3) | $-0.8466 / 3$ | -0.2822 |
| (2,1) | $0.6802 / 3$ | 0.2267 |
| (2,2) | $0.0444 / 3$ | 0.0148 |
| (2,3) | $-0.7247 / 3$ | -0.2416 |

**Final gradient matrix:**

$$\nabla_W J \approx \begin{bmatrix} 0.0445 & 0.2377 & -0.2822 \\ 0.2267 & 0.0148 & -0.2416 \end{bmatrix}$$

---

## 9. Gradient descent — Weight update

**Formula:**

$$W^{(1)} = W^{(0)} - \eta \nabla_W J$$

Given:

$$\eta = 0.5$$

### Step 1: Multiply gradient by learning rate

| Weight position | Gradient | Calculation | Result |
| :--- | :--- | :--- | :--- |
| (1,1) | 0.0445 | $0.5 \times 0.0445$ | 0.02225 |
| (1,2) | 0.2377 | $0.5 \times 0.2377$ | 0.11885 |
| (1,3) | -0.2822 | $0.5 \times -0.2822$ | -0.14110 |
| (2,1) | 0.2267 | $0.5 \times 0.2267$ | 0.11335 |
| (2,2) | 0.0148 | $0.5 \times 0.0148$ | 0.00740 |
| (2,3) | -0.2416 | $0.5 \times -0.2416$ | -0.12080 |

### Step 2: Subtract from initial weights

| Weight | Initial value | Learning-rate $\times$ gradient | Updated value |
| :--- | :--- | :--- | :--- |
| $w_{11}$ | 1 | 0.02225 | 0.97775 |
| $w_{12}$ | 0 | 0.11885 | -0.11885 |
| $w_{13}$ | -1 | -0.14110 | -0.85890 |
| $w_{21}$ | 0 | 0.11335 | -0.11335 |
| $w_{22}$ | 1 | 0.00740 | 0.99260 |
| $w_{23}$ | 0 | -0.12080 | 0.12080 |

**Updated weight matrix:**

$$\boxed{ W^{(1)} \approx \begin{bmatrix} 0.9778 & -0.1188 & -0.8589 \\ -0.1134 & 0.9926 & 0.1208 \end{bmatrix} }$$

---

## 10. Confusion matrix

Using predictions from the initial weights:

| Sample | Actual class | Predicted class | Result |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | Correct |
| 2 | 2 | 2 | Correct |
| 3 | 3 | 1 | Incorrect |

Rows = actual classes; columns = predicted classes.

$$C = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}$$

---

## 11. TP, FP and FN calculations

**Formulas:**

$$TP = C_{kk}$$

$$FP = \text{Column sum} - TP$$

$$FN = \text{Row sum} - TP$$

| Class | Row sum | Column sum | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 2 | 1 | 1 | 0 |
| 2 | 1 | 1 | 1 | 0 | 0 |
| 3 | 1 | 0 | 0 | 0 | 1 |

---

## 12. Precision calculation

**Formula:**

$$\text{Precision} = \frac{TP}{TP + FP}$$

| Class | Calculation | Precision |
| :--- | :--- | :--- |
| 1 | $1 / (1 + 1)$ | 0.500 |
| 2 | $1 / (1 + 0)$ | 1.000 |
| 3 | $0 / (0 + 0)$ | 0* |

*\*Zero is assigned by convention because the denominator is zero.*

---

## 13. Recall calculation

**Formula:**

$$\text{Recall} = \frac{TP}{TP + FN}$$

| Class | Calculation | Recall |
| :--- | :--- | :--- |
| 1 | $1 / (1 + 0)$ | 1.000 |
| 2 | $1 / (1 + 0)$ | 1.000 |
| 3 | $0 / (0 + 1)$ | 0.000 |

---

## 14. F1-score calculation

**Formula:**

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

| Class | Calculation | F1-score |
| :--- | :--- | :--- |
| 1 | $2(0.5 \times 1) / (0.5 + 1)$ | 0.667 |
| 2 | $2(1 \times 1) / (1 + 1)$ | 1.000 |
| 3 | $0$ (zero convention) | 0.000 |

---

## 15. Accuracy calculation

**Formula:**

$$\text{Accuracy} = \frac{\text{Trace}(C)}{N}$$

| Parameter | Calculation | Result |
| :--- | :--- | :--- |
| Correct predictions | $1 + 1 + 0$ | 2 |
| Total samples | $N$ | 3 |
| Accuracy | $2/3$ | 0.667 |
| Accuracy percentage | $0.667 \times 100$ | **66.7%** |

---

## 16. Next iteration — Repeat training

After updating weights, use the new weights for another forward pass.

| Step | Formula | Purpose |
| :--- | :--- | :--- |
| New logits | $Z^{(1)} = XW^{(1)}$ | Calculate new scores |
| New probabilities | $P^{(1)} = \operatorname{softmax}(Z^{(1)})$ | Calculate new probabilities |
| New predictions | $\hat{y}^{(1)} = \arg\max P^{(1)}$ | Select predicted classes |
| New loss | $J(W^{(1)}) = -\frac{1}{N}\sum_i \ln(p_{i,t_i}^{(1)})$ | Measure new loss |
| New error | $D^{(1)} = P^{(1)} - Y$ | Calculate errors |
| New gradient | $\nabla_W J^{(1)} = \frac{1}{N} X^T D^{(1)}$ | Calculate gradients |
| New weights | $W^{(2)} = W^{(1)} - \eta \nabla_W J^{(1)}$ | Update weights again |

---

## 17. Complete formula cheat sheet

| No. | Concept | Formula |
| :--- | :--- | :--- |
| 1 | Input matrix | $X \in \mathbb{R}^{N \times D}$ |
| 2 | Weight matrix | $W \in \mathbb{R}^{D \times K}$ |
| 3 | Logits | $Z = XW$ |
| 4 | Softmax | $p_{ik} = \frac{e^{z_{ik}}}{\sum_j e^{z_{ij}}}$ |
| 5 | Prediction | $\hat{y}_i = \arg\max_k p_{ik}$ |
| 6 | One-hot labels | $Y \in \mathbb{R}^{N \times K}$ |
| 7 | Individual loss | $\mathcal{L}_i = -\ln(p_{i,t_i})$ |
| 8 | Average loss | $J(W) = \frac{1}{N}\sum_i \mathcal{L}_i$ |
| 9 | Error | $D = P - Y$ |
| 10 | Transpose | $X^T \in \mathbb{R}^{D \times N}$ |
| 11 | Gradient | $\nabla_W J = \frac{1}{N} X^T D$ |
| 12 | Weight update | $W_{\text{new}} = W_{\text{old}} - \eta \nabla_W J$ |
| 13 | True Positive | $TP = C_{kk}$ |
| 14 | False Positive | $FP = \text{Column sum} - TP$ |
| 15 | False Negative | $FN = \text{Row sum} - TP$ |
| 16 | Precision | $TP / (TP + FP)$ |
| 17 | Recall | $TP / (TP + FN)$ |
| 18 | F1-score | $2PR / (P + R)$ |
| 19 | Accuracy | $\text{Trace}(C) / N$ |

---

**Numerical note:** All intermediate values above are rounded for readability. For exact reproduction, use full-precision softmax probabilities throughout. Also, the displayed initial loss of 1.2392 is rounded; calculations using unrounded probabilities should be used as the reference for numerical verification.