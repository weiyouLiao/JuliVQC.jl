# **JuliVQC.jl**  | An Efficient Variational Quantum Circuit Simulator for Near-Term Quantum Algorithms

![logo](docs/assets/logo.png)

**JuliVQC.jl** is a high-performance Julia library designed for **variational quantum circuit (VQC)** simulations. Its mission is to provide **seamless support for automatic differentiation** while enabling users to easily build **hybrid quantum-classical algorithms**.

## 🌟 **Key Features**

The core philosophy of **JuliVQC.jl** revolves around three principles:

1. **Transparent implementation**: Built using Julia, a high-performance and dynamic programming language.
2. **Efficiency-focused**: Cache-friendly elementary operations with shared-memory parallelization.
3. **Native automatic differentiation**: Supports both noiseless and noisy quantum circuits.

---

## 🛠️ **Getting Started**

### 📥 Installation 

To get started with **JuliVQC.jl**, you need to install both **JuliVQC.jl** and its dependency **QuantumCircuits.jl**. Note that these packages are not part of Julia's official package registry, so you need to install them directly from GitHub.

---

#### Step 1: Create a Dedicated Julia Environment (Highly Recommended)

We **strongly recommend** creating a dedicated Julia environment for **JuliVQC.jl** to prevent dependency conflicts with other projects. Follow these steps:

1. Open the Julia REPL.
2. Navigate to your project folder (or any desired directory).
3. Activate a new environment:
   ```julia
   pkg> activate .
   ```
   This command creates and activates a new Julia environment in the current folder. A `Project.toml` file will be generated automatically.

---

#### Step 2: Install QuantumCircuits.jl

⚠️ **Important:** The required `QuantumCircuits.jl` package is a custom version different from the official Julia package of the same name. Follow the steps below to avoid conflicts.

Install the custom version of `QuantumCircuits.jl` from the correct GitHub repository using the following command:

```julia
pkg> add https://github.com/weiyouLiao/QuantumCircuits.jl
```

This ensures you're installing the required custom version and avoiding conflicts with the official package.

---

#### Step 3: Install JuliVQC.jl

Once the correct version of `QuantumCircuits.jl` is installed, proceed to install **JuliVQC.jl** from its GitHub repository:

```julia
pkg> add https://github.com/weiyouLiao/JuliVQC.jl
```

---

#### Step 4: Verify Installation

After completing the above steps, verify the installation by loading **JuliVQC.jl** in the Julia REPL:

```julia
using JuliVQC
```

If no errors occur, the installation was successful, and you're ready to start building variational quantum circuits!

---

## 🚀 **Quick Start**

Here’s how you can get started with **JuliVQC.jl**:

### Example 1: Create a Two-Qubit Bell State
```julia
using JuliVQC
using QuantumCircuits

# Create a 2-qubit quantum state
state = StateVector(2)

# Define a quantum circuit
circuit = QCircuit()
push!(circuit, HGate(1))         # Add a Hadamard gate on qubit 1
push!(circuit, CNOTGate(1, 2))   # Add a CNOT gate between qubits 1 and 2

# Apply the circuit to the quantum state
apply!(circuit, state)

# Perform quantum measurement
i, prob = measure!(state, 1)
println("Probability of qubit 1 being in state $i is $prob.")
```

---

### Example 2: Build and Optimize a Variational Quantum Circuit

Constructing a variational quantum circuit is just as easy:

```julia
using JuliVQC
using QuantumCircuits
using Zygote

L = 3  # Number of qubits
state = StateVector(L)

# Define a variational quantum circuit
circuit = QCircuit()
for i in 1:L
    push!(circuit, RzGate(i, rand(), isparas=true))
    push!(circuit, RyGate(i, rand(), isparas=true))
    push!(circuit, RzGate(i, rand(), isparas=true))
end

for depth in 1:2
    for i in 1:L-1
        push!(circuit, CNOTGate(i, i+1))
    end
    for i in 1:L
        push!(circuit, RzGate(i, rand(), isparas=true))
        push!(circuit, RxGate(i, rand(), isparas=true))
        push!(circuit, RzGate(i, rand(), isparas=true))
    end
end

# Define a target state
target_state = StateVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])

# Define a loss function
loss(c) = distance(target_state, c * state)

# Compute the gradient of the loss function
v = loss(circuit)
grad = gradient(loss, circuit)
```

---

## 📄 **Documentation**

### Online Documentation 🌐

The full documentation for **JuliVQC.jl** is hosted online via **GitHub Pages**. Visit the link below for detailed explanations, tutorials:

👉 [Online Documentation](https://weiyouliao.github.io/JuliVQC.jl/)

---

### Local Deployment 🖥️

If you'd like to view the documentation locally, you can do so by building and serving it on your machine:

1. Clone this repository:
   ```bash
   git clone https://github.com/weiyouLiao/JuliVQC.jl.git
   cd JuliVQC.jl
   ```

2. Install Julia dependencies in the `docs` environment:
   ```julia
   cd docs
   julia --project=.
   using Pkg
   Pkg.instantiate()
   ```

3. Build the documentation:
   ```julia
   julia make.jl
   ```

4. Open the `docs/build/index.html` file in your browser to view the locally built documentation.

## 📰 **Paper**

This library is introduced and detailed in the our paper:

 **[JuliVQC: an Efficient Variational Quantum Circuit Simulator for Near-Term Quantum Algorithms](https://arxiv.org/abs/2406.19212))**
This paper provides a comprehensive description of the methodology behind **JuliVQC.jl**, including its design principles, and implementation details.

------

## 📊 **Benchmarks**

Benchmark results comparing **JuliVQC.jl** with other quantum circuit simulators are available in a separate repository:

👉 [Benchmark Repository](https://github.com/weiyouLiao/JuliVQC-benchmark)