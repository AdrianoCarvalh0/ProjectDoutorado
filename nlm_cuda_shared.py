from numba import cuda, float32
import math
import skimage.io
import skimage.color

# Ajuste o tamanho máximo da memória compartilhada conforme seu blockDim e pad
MAX_SHARED_MEM = 48 * 48  # exemplo: blockDim=(32,32) + padding

@cuda.jit
def nlm_kernel_shared(img_n, output, m, n, f, t, h, padded_width):
    # Índices globais (pixel a processar)
    i, j = cuda.grid(2)

    # Dimensões do bloco
    Bx = cuda.blockDim.x
    By = cuda.blockDim.y

    # Padding da vizinhança + busca
    pad = f + t

    # Dimensões da memória compartilhada (bloco + padding)
    sh_width = Bx + 2 * pad
    sh_height = By + 2 * pad

    # Aloca shared memory 1D para o patch + padding
    sh_img = cuda.shared.array(shape=MAX_SHARED_MEM, dtype=float32)

    # Base na imagem global para carregar shared memory (top-left do bloco com padding)
    base_i = cuda.blockIdx.y * By + f - pad
    base_j = cuda.blockIdx.x * Bx + f - pad

    # Cada thread carrega pixels para a memória compartilhada, em tiles
    for y in range(cuda.threadIdx.y, sh_height, By):
        for x in range(cuda.threadIdx.x, sh_width, Bx):
            img_i = base_i + y
            img_j = base_j + x

            # Clamp para borda da imagem com padding
            ii = max(0, min(img_i, m + 2 * f - 1))
            jj = max(0, min(img_j, n + 2 * f - 1))

            # Carrega o pixel na memória compartilhada (indexação 1D)
            sh_img[y * sh_width + x] = img_n[ii, jj]

    # Sincroniza todas as threads para garantir que shared memory esteja carregada
    cuda.syncthreads()

    # Sai se fora da área válida da imagem original (sem padding)
    if i >= m or j >= n:
        return

    # Índices locais dentro da memória compartilhada para o pixel atual
    local_i = cuda.threadIdx.y + pad
    local_j = cuda.threadIdx.x + pad

    NL = 0.0  # soma ponderada do valor dos pixels
    Z = 0.0   # soma dos pesos (normalização)

    # Limites da vizinhança de busca na shared memory (janela centrada no pixel)
    rmin = max(local_i - t, pad)
    rmax = min(local_i + t, By + pad - 1)
    smin = max(local_j - t, pad)
    smax = min(local_j + t, Bx + pad - 1)

    # Loop de comparação da vizinhança (patch)
    for r in range(rmin, rmax + 1):
        for s in range(smin, smax + 1):
            d2 = 0.0  # distância ao quadrado entre patches

            for u in range(-f, f + 1):
                for v in range(-f, f + 1):
                    a = sh_img[(local_i + u) * sh_width + (local_j + v)]
                    b = sh_img[(r + u) * sh_width + (s + v)]
                    diff = a - b
                    d2 += diff * diff

            sij = math.exp(-d2 / (h * h))
            Z += sij
            NL += sij * sh_img[r * sh_width + s]

    # Evita divisão por zero
    if Z > 0:
        output[i * n + j] = NL / Z
    else:
        output[i * n + j] = 0.0


import numpy as np
from numba import cuda

# Parâmetros (exemplo)
f = 3   # half patch size
t = 10  # search window radius
h = 0.1 # filtro h

img_path = 'extras/images/ct2.png'
img = skimage.io.imread(img_path)
if len(img.shape) > 2:
    img = skimage.color.rgb2gray(img)
    img = 255 * img

sigma = 10

m, n = img.shape

# Pad a imagem antes e envie para GPU
pad = f + t
img_padded = np.pad(img, pad, mode='reflect').astype(np.float32)

d_img = cuda.to_device(img_padded)
d_output = cuda.device_array(m * n, dtype=np.float32)

# Dimensões do bloco e grade
block_dim = (32, 16)   # Exemplo
grid_dim = ((n + block_dim[0] - 1) // block_dim[0],
            (m + block_dim[1] - 1) // block_dim[1])

shared_size = (block_dim[0] + 2 * pad) * (block_dim[1] + 2 * pad) * 4  # bytes (float32=4bytes)

nlm_kernel_shared[grid_dim, block_dim, shared_size](d_img, d_output, m, n, f, t, h, img_padded.shape[1])


output = d_output.copy_to_host()
output = output.reshape((m, n))
