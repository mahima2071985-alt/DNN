# RNN -- Lesson 1: Introduction and Fundamentals

**Course:** BITS Pilani -- AIML ZG511: Deep Neural Networks  
**Syllabus alignment:** Session 9 -- Recurrent Neural Networks  
**Level:** Beginner  
**Focus:** Concepts and step-by-step mathematics  

---

## 1. Learning objectives

By the end of this lesson, you should be able to:

1. Explain what sequence data is.
2. Explain why Recurrent Neural Networks (RNNs) are useful.
3. Distinguish a Feedforward Neural Network from an RNN.
4. Explain the role of a hidden state.
5. Identify the input, hidden state, weights, bias, and activation function in a basic RNN.
6. Calculate hidden states over multiple time steps using a simple RNN equation.
7. Distinguish a hidden state from the network output.

---

## 2. What is sequence data?

Sequence data is data arranged in a meaningful order. Earlier elements in the sequence may help us understand or predict later elements.

### Example: daily temperature

| Day | Temperature |
| :--- | :--- |
| Day 1 | 28°C |
| Day 2 | 30°C |
| Day 3 | 32°C |
| Day 4 | 34°C |
| Day 5 | ? |

We may use the temperatures from earlier days to predict the temperature on Day 5.

Examples of sequence data include:

| Application | Sequence |
| :--- | :--- |
| Weather prediction | Daily temperatures |
| Stock prediction | Daily stock prices |
| Speech recognition | Audio signals over time |
| Language processing | Words in a sentence |
| Video processing | Ordered video frames |

**Key idea:** Sequence data is ordered data in which previous information can be useful for understanding or predicting what comes next.

---

## 3. Why do we need an RNN?

A basic Feedforward Neural Network processes information from input towards output. It does not have a recurrent connection that carries its previous hidden representation into the next time step.

An RNN is designed to process sequential inputs. It passes information from one time step to the next through a **hidden state**.

### Feedforward Neural Network

A simplified flow is:

`Input → Hidden Layer → Output`

There is no recurrent connection in this basic architecture.

### Recurrent Neural Network

A simplified sequence of steps is:

$x_1 \to h_1 \to h_2 \to h_3$

Each hidden state is calculated using the current input and the previous hidden state:

- **Step 1** uses the first input and the initial hidden state.
- **Step 2** uses the second input and the hidden state from Step 1.
- **Step 3** uses the third input and the hidden state from Step 2.

This is called a **recurrent connection**.

---

## 4. What is a hidden state?

The hidden state is an internal numerical representation that carries information from previous time steps.

Consider the sentence:

> *I love machine learning*

The words arrive in sequence:

| Time step | Input |
| :--- | :--- |
| 1 | I |
| 2 | love |
| 3 | machine |
| 4 | learning |

When the RNN processes the word `I`, it produces a hidden state. When it processes `love`, it uses both the current input and the previous hidden state.

The hidden state is not a literal copy of the earlier word or sentence. It is a numerical representation learned by the network.

**Definition:** A hidden state is an internal representation that carries information from previous time steps.

---

## 5. Basic RNN mathematical equation

The basic hidden-state equation is:

$$h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

Where:

| Symbol | Meaning |
| :--- | :--- |
| $x_t$ | Current input at time step $t$ |
| $h_{t-1}$ | Hidden state from the previous time step |
| $h_t$ | Current hidden state |
| $W_{xh}$ | Input-to-hidden weight |
| $W_{hh}$ | Hidden-to-hidden (recurrent) weight |
| $b_h$ | Hidden-layer bias |
| $f$ | Activation function |
| $t$ | Current time step |

For example:
- $t=1$: first time step
- $t=2$: second time step
- $t=3$: third time step

### Shared weights

In a basic RNN, the same weight matrices are reused at every time step. This is called **weight sharing**.

The network does not create a completely new set of recurrent weights for every position in the sequence.

---

## 6. Worked numerical example: calculating hidden states

We will use a simple one-dimensional RNN.

### Given data

| Time step | Input ($x_t$) |
| :--- | :--- |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |

**Parameters:**

$$W_{xh} = 0.5$$
$$W_{hh} = 0.5$$
$$b_h = 0$$
$$h_0 = 0$$

**Activation function:**

$$\text{ReLU}(z) = \max(0,z)$$

Therefore:

$$h_t = \text{ReLU}(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

This is a deliberately small example to make the recurrent calculation easy to follow. It is not a complete trained forecasting model.

---

### Step 1: Calculate $h_1$

Use the first input and initial hidden state:

$$h_1 = \text{ReLU}(W_{xh}x_1 + W_{hh}h_0 + b_h)$$

Substitute the values:

$$h_1 = \text{ReLU}(0.5(2) + 0.5(0) + 0)$$

Calculate the multiplications:

$$h_1 = \text{ReLU}(1 + 0 + 0)$$
$$h_1 = \text{ReLU}(1)$$
$$\mathbf{h_1 = 1}$$

---

### Step 2: Calculate $h_2$

Use the second input and the previous hidden state ($h_1 = 1$):

$$h_2 = \text{ReLU}(W_{xh}x_2 + W_{hh}h_1 + b_h)$$

Substitute:

$$h_2 = \text{ReLU}(0.5(4) + 0.5(1) + 0)$$
$$h_2 = \text{ReLU}(2 + 0.5)$$
$$h_2 = \text{ReLU}(2.5)$$
$$\mathbf{h_2 = 2.5}$$

---

### Step 3: Calculate $h_3$

Use the third input and the previous hidden state ($h_2 = 2.5$):

$$h_3 = \text{ReLU}(W_{xh}x_3 + W_{hh}h_2 + b_h)$$

Substitute:

$$h_3 = \text{ReLU}(0.5(6) + 0.5(2.5) + 0)$$
$$h_3 = \text{ReLU}(3 + 1.25)$$
$$h_3 = \text{ReLU}(4.25)$$
$$\mathbf{h_3 = 4.25}$$

---

### Final hidden-state results

| Time step | Input ($x_t$) | Previous state | Current state |
| :--- | :--- | :--- | :--- |
| 1 | 2 | $h_0 = 0$ | $h_1 = 1$ |
| 2 | 4 | $h_1 = 1$ | $h_2 = 2.5$ |
| 3 | 6 | $h_2 = 2.5$ | $h_3 = 4.25$ |

**Important observation:**
- To calculate $h_2$, we used $h_1$.
- To calculate $h_3$, we used $h_2$.

This is the recurrent connection: the previous hidden state is used when calculating the next hidden state.

---

## 7. Hidden state versus output

The hidden state is not necessarily the final prediction.

A basic output-layer equation is:

$$y_t = W_{hy}h_t + b_y$$

Where:

| Symbol | Meaning |
| :--- | :--- |
| $y_t$ | Output at time step $t$ |
| $W_{hy}$ | Hidden-to-output weight |
| $h_t$ | Current hidden state |
| $b_y$ | Output bias |

### Worked output example

Suppose:

$$W_{hy} = 2, \quad b_y = 0$$

At the third time step, we calculated:

$$h_3 = 4.25$$

Then:

$$y_3 = 2(4.25) + 0$$
$$\mathbf{y_3 = 8.5}$$

Therefore:
- **Final hidden state:** $h_3 = 4.25$
- **Output:** $y_3 = 8.5$

They are different quantities.

---

## 8. A second practice problem

Given:

$$x_1 = 3, \quad x_2 = 5$$
$$W_{xh} = 0.4$$
$$W_{hh} = 0.2$$
$$b_h = 0, \quad h_0 = 0$$

Activation function: **ReLU**.

Calculate $h_1$ and $h_2$.

### Step 1: Calculate $h_1$

$$h_1 = \text{ReLU}(0.4(3) + 0.2(0) + 0)$$
$$h_1 = \text{ReLU}(1.2)$$
$$\mathbf{h_1 = 1.2}$$

### Step 2: Calculate $h_2$

Use the previous hidden state ($h_1 = 1.2$):

$$h_2 = \text{ReLU}(0.4(5) + 0.2(1.2) + 0)$$
$$h_2 = \text{ReLU}(2 + 0.24)$$
$$\mathbf{h_2 = 2.24}$$

### Correct answers

| Hidden state | Answer |
| :--- | :--- |
| $h_1$ | 1.2 |
| $h_2$ | 2.24 |

---

## 9. Important exam revision table

| Concept | Remember |
| :--- | :--- |
| **Sequence data** | Data arranged in a meaningful order |
| **RNN** | Neural network designed to process sequential data |
| **Recurrent connection** | Carries the previous hidden state to the next step |
| **Hidden state** | Internal representation carrying information across time steps |
| **Time step** | One position in a sequence |
| **Shared weights** | The same RNN weights are reused at each time step |
| **Output layer** | Converts a hidden state into an output |

### Main formula to remember

$$\mathbf{h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h)}$$

Output equation:

$$\mathbf{y_t = W_{hy}h_t + b_y}$$

---

## 10. Knowledge check

Try answering these questions without looking at the explanations.

1. What is sequence data?
2. Why is an RNN useful for sequential information?
3. What does the hidden state represent?
4. What does $t$ represent in an RNN equation?
5. What is a recurrent connection?
6. Are the RNN weights shared across time steps?
7. If $h_1=1$, $x_2=4$, $W_{xh}=0.5$, $W_{hh}=0.5$, and $b_h=0$, what is $h_2$ using ReLU?
8. What is the difference between a hidden state and an output?

---

## 11. Lesson 1 summary

The most important idea in this lesson is that an RNN uses the current input and the previous hidden state to calculate the current hidden state.

$$\text{Current hidden state} = f(\text{Current input contribution} + \text{Previous state contribution} + \text{Bias})$$

Do not confuse the hidden state with the final output. The hidden state carries information; the output layer produces the network's prediction.

---

**Next lesson -- RNN Lesson 2**

- RNN architecture in detail
- Unrolling an RNN through time
- Input, hidden, and output dimensions
- Matrix-based RNN equations
- More numerical examples
- A basic Python implementation

**Study tip:** Understand the hidden-state equation and practise calculating each time step before moving to Backpropagation Through Time (BPTT).