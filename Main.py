import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
input_image = cv2.imread('Assets/Image/Picture2.png', cv2.IMREAD_GRAYSCALE)
M, N = input_image.shape

# Bước 1: Chuyển kích thước ảnh
P, Q = 2 * M, 2 * N
fp = np.zeros((P, Q), dtype=np.float32)
fp[:M, :N] = input_image

# Bước 2: Tạo ảnh mới Fp
Fp = np.multiply(fp, (-1) ** np.add.outer(np.arange(P), np.arange(Q)))

# Bước 3: Tính DFT của Fp
F = np.fft.fft2(Fp)

# Bước 4: Tạo bộ lọc Butterworth thông cao
D0 = 100  # Thay đổi D0 theo ý muốn
n = 2     # Thay đổi n theo ý muốn
u = np.arange(P)
v = np.arange(Q)
U, V = np.meshgrid(u, v)
U = U - P/2
V = V - Q/2
D = np.sqrt(U**2 + V**2)
H = 1 / (1 + (D0 / D) ** (2 * n))

# Bước 5: Áp dụng bộ lọc thông cao
G = np.multiply(F, H)


# Bước 6: Biến đổi ngược IDFT của G
gp = np.real(np.fft.ifft2(G))
gp = np.multiply(gp, (-1) ** np.add.outer(np.arange(P), np.arange(Q)))

# Bước 7: Lấy vùng MxN từ ảnh gp
g = gp[:M, :N]

# Hiển thị ảnh đầu ra bằng matplotlib
plt.subplot(121), plt.imshow(input_image, cmap='gray'), plt.title('Ảnh đầu vào')
plt.subplot(122), plt.imshow(g, cmap='gray'), plt.title('Ảnh đầu ra')
plt.show()
