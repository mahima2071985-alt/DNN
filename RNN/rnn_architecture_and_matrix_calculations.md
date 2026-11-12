# RNN – Lesson 2: Architecture, Unrolling and Matrix Calculations



## 2. Basic RNN architecture

A basic RNN has three main parts:

1. **Input ($x_t$)** – the data entering the RNN at time step $t$.
2. **Hidden state ($h_t$)** – information carried from one time step to the next.
3. **Output ($y_t$)** – the prediction produced by the RNN.

At each time step, the RNN uses the current input $x_t$ and previous hidden state $h_{t-1}$ to calculate the current hidden state $h_t$.

$$h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

| Symbol | Meaning |
|---|---|
| $x_t$ | Current input |
| $h_{t-1}$ | Previous hidden state |
| $h_t$ | Current hidden state |
| $W_{xh}$ | Input-to-hidden weight matrix |
| $W_{hh}$ | Hidden-to-hidden (recurrent) weight matrix |
| $b_h$ | Hidden-layer bias |
| $f$ | Activation function, such as $\text{ReLU}$ or $\tanh$ |

**Main idea:** Current input + information from the previous time step = new hidden state.

---

## 3. Unrolling an RNN through time

An RNN processes a sequence one time step at a time:

- **Time step 1:** $x_1$ and initial hidden state $h_0$ produce $h_1$.
- **Time step 2:** $x_2$ and $h_1$ produce $h_2$.
- **Time step 3:** $x_3$ and $h_2$ produce $h_3$.

This repeated view of the RNN cell across time is called **unrolling through time**.

$$h_1 = f(W_{xh}x_1 + W_{hh}h_0 + b_h)$$

$$h_2 = f(W_{xh}x_2 + W_{hh}h_1 + b_h)$$

$$h_3 = f(W_{xh}x_3 + W_{hh}h_2 + b_h)$$

---

## 4. Weight sharing

The RNN reuses the **same weight matrices** at every time step:
- $W_{xh}$ is reused for every input.
- $W_{hh}$ is reused for every previous hidden state.

It does not create a new set of weights for each time step. This is called **weight sharing**.

---

## 5. Matrix dimensions

Consider an RNN with 2 input features, 2 hidden neurons, and 1 output. Inputs and hidden states are column vectors.

| Quantity | Dimension | Explanation |
|---|---:|---|
| $x_t$ | $2 \times 1$ | 2 input features |
| $h_t$ | $2 \times 1$ | 2 hidden values |
| $W_{xh}$ | $2 \times 2$ | Maps 2 input values to 2 hidden values |
| $W_{hh}$ | $2 \times 2$ | Maps 2 previous hidden values to 2 current hidden values |
| $b_h$ | $2 \times 1$ | One bias for each hidden neuron |
| $W_{hy}$ | $1 \times 2$ | Maps 2 hidden values to 1 output |
| $b_y$ | $1 \times 1$ | Output bias |

### Matrix multiplication rule

$$(m \times n)(n \times p) = (m \times p)$$

The **inside dimensions must match**. For example:

$$(2 \times 2)(2 \times 1) = (2 \times 1)$$

So $W_{xh} x_t$ produces a $2 \times 1$ vector.

---

## 6. Worked numerical example

Given:

$$x_1 = \begin{bmatrix}1\\2\end{bmatrix}, \quad h_0 = \begin{bmatrix}0\\0\end{bmatrix}$$

$$W_{xh} = \begin{bmatrix}0.5 & 0.2\\0.1 & 0.4\end{bmatrix}, \quad W_{hh} = \begin{bmatrix}0.3 & 0.1\\0.2 & 0.5\end{bmatrix}$$

Assume $b_h = \begin{bmatrix}0\\0\end{bmatrix}$ and the activation function is $\text{ReLU}$.

### Step 1: Calculate $W_{xh}x_1$

$$W_{xh}x_1 = \begin{bmatrix} (0.5 \times 1) + (0.2 \times 2) \\ (0.1 \times 1) + (0.4 \times 2) \end{bmatrix} = \begin{bmatrix}0.9\\0.9\end{bmatrix}$$

### Step 2: Calculate $W_{hh}h_0$

$$W_{hh}h_0 = \begin{bmatrix} (0.3 \times 0) + (0.1 \times 0) \\ (0.2 \times 0) + (0.5 \times 0) \end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}$$

### Step 3: Add and apply $\text{ReLU}$

$$h_1 = \text{ReLU}\left(\begin{bmatrix}0.9\\0.9\end{bmatrix} + \begin{bmatrix}0\\0\end{bmatrix}\right)$$

$$\boxed{h_1 = \begin{bmatrix}0.9\\0.9\end{bmatrix}}$$

### Step 4: Calculate the next hidden state

Given:

$$x_2 = \begin{bmatrix}2\\1\end{bmatrix}$$

Current-input contribution:

$$W_{xh}x_2 = \begin{bmatrix} (0.5 \times 2) + (0.2 \times 1) \\ (0.1 \times 2) + (0.4 \times 1) \end{bmatrix} = \begin{bmatrix}1.2\\0.6\end{bmatrix}$$

Previous-hidden-state contribution:

$$W_{hh}h_1 = \begin{bmatrix} (0.3 \times 0.9) + (0.1 \times 0.9) \\ (0.2 \times 0.9) + (0.5 \times 0.9) \end{bmatrix} = \begin{bmatrix}0.36\\0.63\end{bmatrix}$$

Add both contributions and apply $\text{ReLU}$:

$$h_2 = \text{ReLU}\left(\begin{bmatrix}1.2\\0.6\end{bmatrix} + \begin{bmatrix}0.36\\0.63\end{bmatrix}\right)$$

$$\boxed{h_2 = \begin{bmatrix}1.56\\1.23\end{bmatrix}}$$

### Step 5: Calculate an output

Given:

$$W_{hy} = \begin{bmatrix}0.6 & 0.4\end{bmatrix}, \quad b_y = 0.1$$

Output equation:

$$y_t = W_{hy}h_t + b_y$$

At time step 2:

$$y_2 = (0.6 \times 1.56) + (0.4 \times 1.23) + 0.1$$

$$= 0.936 + 0.492 + 0.1$$

$$\boxed{y_2 = 1.528}$$

---

## 7. Practice calculation from the lesson

Given:

$$W_{xh} = \begin{bmatrix}0.2 & 0.3\\0.4 & 0.1\end{bmatrix}, \quad x_1 = \begin{bmatrix}2\\1\end{bmatrix}$$

$$W_{hh} = \begin{bmatrix}0.5 & 0.2\\0.1 & 0.3\end{bmatrix}, \quad h_0 = \begin{bmatrix}0\\0\end{bmatrix}$$

Bias is zero and activation is $\text{ReLU}$.

### Calculate $h_1$

$$W_{xh}x_1 = \begin{bmatrix} (0.2 \times 2) + (0.3 \times 1) \\ (0.4 \times 2) + (0.1 \times 1) \end{bmatrix} = \begin{bmatrix}0.7\\0.9\end{bmatrix}$$

Since $h_0$ is zero, $W_{hh}h_0 = \begin{bmatrix}0\\0\end{bmatrix}$.

$$\boxed{h_1 = \begin{bmatrix}0.7\\0.9\end{bmatrix}}$$

### Calculate $h_2$

Now use:

$$x_2 = \begin{bmatrix}1\\2\end{bmatrix}$$

Current-input contribution:

$$W_{xh}x_2 = \begin{bmatrix} (0.2 \times 1) + (0.3 \times 2) \\ (0.4 \times 1) + (0.1 \times 2) \end{bmatrix} = \begin{bmatrix}0.8\\0.6\end{bmatrix}$$

Previous-hidden-state contribution:

$$W_{hh}h_1 = \begin{bmatrix} (0.5 \times 0.7) + (0.2 \times 0.9) \\ (0.1 \times 0.7) + (0.3 \times 0.9) \end{bmatrix} = \begin{bmatrix}0.53\\0.34\end{bmatrix}$$

Add the two vectors:

$$h_2 = \text{ReLU}\left(\begin{bmatrix}0.8\\0.6\end{bmatrix} + \begin{bmatrix}0.53\\0.34\end{bmatrix}\right)$$

$$\boxed{h_2 = \begin{bmatrix}1.33\\0.94\end{bmatrix}}$$

### Why do we add these two terms?

- $W_{xh}x_2$ contributes information from the **current input**.
- $W_{hh}h_1$ contributes information carried from the **previous hidden state**.

Together, they calculate the new hidden state $h_2$.

---

## 8. Common mistakes to avoid

1. **Multiplying corresponding entries only:** Matrix multiplication requires multiplying a row by a column and adding the products.
2. **Ignoring vector dimensions:** Check that the inner dimensions match before multiplying.
3. **Using the wrong previous hidden state:** To calculate $h_2$, use $h_1$; to calculate $h_3$, use $h_2$.
4. **Forgetting weight sharing:** The same $W_{xh}$ and $W_{hh}$ are reused at each time step.
5. **Forgetting the activation function:** Add the terms and bias first, then apply the specified activation function.
6. **Decimal-place mistakes:** For example, $0.2 \times 3 = 0.6$, not $0.06$.

---

## 9. Quick revision

| Question | Short answer |
|---|---|
| What does $x_t$ represent? | Current input |
| What does $h_{t-1}$ represent? | Previous hidden state |
| What is $h_t$? | Current hidden state |
| What does $W_{xh}$ do? | Transforms current input |
| What does $W_{hh}$ do? | Transforms previous hidden state |
| What is unrolling? | Showing the RNN repeated across time steps |
| What is weight sharing? | Reusing the same weights at every time step |
| Why add the two matrix products? | To combine current input information and previous hidden-state information |

---

