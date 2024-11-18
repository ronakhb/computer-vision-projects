import numpy as np
import matplotlib.pyplot as plt

def sobel_kernel(size=5):
    sobel_x = np.array([[2, 2, 4, 2, 2],
                        [1, 1, 2, 1, 1],
                        [0, 0, 0, 0, 0],
                        [-1, -1, -2, -1, -1],
                        [-2, -2, -4, -2, -2]])
    sobel_y = np.array([[2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2],
                        [4, 2, 0, -2, -4],
                        [2, 1, 0, -1, -2],
                        [2, 1, 0, -1, -2]])
    gaussian_kernel = np.array([[1, 2, 4, 2, 1],
                                [2, 4, 8, 4, 2],
                                [4, 8, 16, 8, 4],
                                [2, 4, 8, 4, 2],
                                [1, 2, 4, 2, 1]])
    
    sobel_gaussian = sobel_x @ sobel_y
    return sobel_gaussian, sobel_y

def generate_gaussian():
    gaussian_kernel = np.array([[1, 2, 4, 2, 1],
                                [2, 4, 8, 4, 2],
                                [4, 8, 16, 8, 4],
                                [2, 4, 8, 4, 2],
                                [1, 2, 4, 2, 1]])
    return gaussian_kernel

def gabor_kernel(size=5, sigma=1, theta=0, Lambda=5, psi=0, gamma=1):
    # Create grid
    x, y = np.meshgrid(np.arange(-size//2 + 1, size//2 + 1), np.arange(-size//2 + 1, size//2 + 1))
    
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    
    # Gabor function
    gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
    sine_wave = np.cos(2 * np.pi / Lambda * x_theta + psi)
    gabor = gaussian * sine_wave
    
    return gabor

def sobel_derivative(sobel_kernel):
    sobel_x, sobel_y = sobel_kernel
    derivative_sobel_x = np.gradient(sobel_x)[0]
    derivative_sobel_y = np.gradient(sobel_y)[1]
    return derivative_sobel_x, derivative_sobel_y

# Create Sobel kernel
sobel_x, sobel_y = sobel_kernel()

# Create Gabor kernel
gabor = gabor_kernel()

gaussian = generate_gaussian()


# Create derivative of Sobel kernel
derivative_sobel_x, derivative_sobel_y = sobel_derivative((sobel_x, sobel_y))

# Set colormap
cmap = 'hot'

# Visualize kernels
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(sobel_x, cmap=cmap)
plt.title('Sobel X Kernel')

plt.subplot(2, 3, 2)
plt.imshow(sobel_y, cmap=cmap)
plt.title('Sobel Y Kernel')

plt.subplot(2, 3, 3)
plt.imshow(gabor, cmap=cmap)
plt.title('Gabor Kernel')

plt.subplot(2, 3, 4)
plt.imshow(gaussian, cmap=cmap)
plt.title('Gaussian Kernel')

plt.subplot(2, 3, 5)
plt.imshow(derivative_sobel_x, cmap=cmap)
plt.title('Derivative of Sobel X Kernel')

plt.subplot(2, 3, 6)
plt.imshow(derivative_sobel_y, cmap=cmap)
plt.title('Derivative of Sobel Y Kernel')

plt.tight_layout()
plt.show()
