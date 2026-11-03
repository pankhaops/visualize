import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 100)


linear = x
binary_step = np.where(x >= 0, 1, 0)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)


plt.figure(figsize=(12, 8))

# 1. Linear
plt.subplot(3, 2, 1)
plt.plot(x, linear, color='blue')
plt.title("Linear Activation")
plt.grid(True)

# 2. Binary Step
plt.subplot(3, 2, 2)
plt.plot(x, binary_step, color='red')
plt.title("Binary Step Activation")
plt.grid(True)

# 3. Sigmoid
plt.subplot(3, 2, 3)
plt.plot(x, sigmoid, color='green')
plt.title("Sigmoid Activation")
plt.grid(True)

# 4. Tanh
plt.subplot(3, 2, 4)
plt.plot(x, tanh, color='purple')
plt.title("Tanh Activation")
plt.grid(True)

# 5. ReLU
plt.subplot(3, 2, 5)
plt.plot(x, relu, color='orange')
plt.title("ReLU Activation")
plt.grid(True)

plt.tight_layout()
plt.show()


plt.tight_layout()
plt.show()


sample_input = np.array([-3, -1, 0, 1, 3])

print("Sample Input :", sample_input)

print("\nLinear:")
print(sample_input)

print("\nBinary Step:")
print(np.where(sample_input >= 0, 1, 0))

print("\nSigmoid:")
print(np.round(1 / (1 + np.exp(-sample_input)), 4))

print("\nTanh:")
print(np.round(np.tanh(sample_input), 4))

print("\nReLU:")
print(np.maximum(0, sample_input))
