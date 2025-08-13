#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GEONLM com CUDA (CuPy + cuML + cuGraph)
- k-NN na GPU (cuML)
- Dijkstra/SSSP na GPU (cuGraph)
- Tensores no device (CuPy)

Observação: o algoritmo mantém a estrutura pixel-a-pixel e monta um grafo kNN
para cada janela de busca, como no seu código. O ganho principal vem de kNN e SSSP
rodando na GPU. Para acelerar ainda mais, seria necessário reestruturar para lotes.
"""

import warnings, time
import numpy as np
import cupy as cp
import skimage.io, skimage.color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import cudf
import cugraph
from cuml.neighbors import NearestNeighbors

warnings.simplefilter(action='ignore')

# -----------------------------
# Utilidades em GPU
# -----------------------------

def pad_symmetric_gpu(img_cp: cp.ndarray, f: int) -> cp.ndarray:
    # CuPy tem pad com 'symmetric'
    return cp.pad(img_cp, ((f, f), (f, f)), mode='symmetric')

def extract_window_patches_gpu(img_n: cp.ndarray, i: int, j: int, f: int, t: int, m: int, n: int):
    """
    Extrai todos os patches (flatten) dentro da janela de busca do pixel (i,j)
    - i,j são índices relativos à imagem sem padding (0..m-1, 0..n-1)
    Retorna:
      dataset_cp: (num_elem, patch_size) em GPU (float32)
      source_idx: índice do patch central dentro do dataset
      pixels_busca_cp: intensidades (num_elem,) dos centros dos patches (para ponderação)
    """
    im = i + f
    jn = j + f

    rmin = max(im - t, f)
    rmax = min(im + t, m + f)
    smin = max(jn - t, f)
    smax = min(jn + t, n + f)

    patch_size = (2 * f + 1) * (2 * f + 1)
    num_elem = (rmax - rmin) * (smax - smin)

    dataset_cp = cp.empty((num_elem, patch_size), dtype=cp.float32)
    pixels_busca_cp = cp.empty((num_elem,), dtype=cp.float32)

    k = 0
    source_idx = -1

    # Observação: laços Python ainda existem, mas cada slice/manipulação é no device
    for r in range(rmin, rmax):
        for s in range(smin, smax):
            W = img_n[r - f:r + f + 1, s - f:s + f + 1]
            dataset_cp[k, :] = W.astype(cp.float32).ravel()
            pixels_busca_cp[k] = img_n[r, s]
            if r == im and s == jn:
                source_idx = k
            k += 1

    return dataset_cp, source_idx, pixels_busca_cp


def knn_graph_edges_gpu(dataset_cp: cp.ndarray, nn: int):
    """
    Usa cuML NearestNeighbors para obter vizinhos e distâncias na GPU.
    Retorna um cudf.DataFrame de arestas: ['src','dst','weight']
    """
    n_samples = dataset_cp.shape[0]

    # cuML aceita cupy arrays diretamente
    knn = NearestNeighbors(n_neighbors=nn, metric='euclidean', output_type='cupy')
    knn.fit(dataset_cp)

    distances_cp, indices_cp = knn.kneighbors(dataset_cp)

    # Monta lista de arestas (u -> v) com pesos = distância
    src_cp = cp.repeat(cp.arange(n_samples, dtype=cp.int32), nn)
    dst_cp = indices_cp.reshape(-1).astype(cp.int32)
    w_cp = distances_cp.reshape(-1).astype(cp.float32)

    # Constrói cudf DataFrame diretamente de arrays no device
    edges_gdf = cudf.DataFrame({
        'src': src_cp,
        'dst': dst_cp,
        'weight': w_cp
    })
    return edges_gdf


def sssp_gpu(edges_gdf: cudf.DataFrame, source_idx: int):
    """
    Roda SSSP (Dijkstra) em GPU via cuGraph.
    edges_gdf precisa ter colunas: 'src','dst','weight'
    Retorna cudf com colunas ['vertex','distance','predecessor']
    """
    G = cugraph.Graph(directed=True)  # grafo direcionado (kNN é naturalmente dirigido)
    G.from_cudf_edgelist(edges_gdf, source='src', destination='dst', edge_attr='weight', renumber=False)
    df_sssp = cugraph.sssp(G, source=source_idx, weight='weight')
    return df_sssp


def geonlm_pixel_gpu(img_n: cp.ndarray, i: int, j: int, f: int, t: int, h: float, nn: int, m: int, n: int) -> float:
    """
    Computa o valor GEONLM para um único pixel (i,j) usando KNN + SSSP na GPU.
    Retorna escalar float (no host).
    """
    dataset_cp, source_idx, pixels_busca_cp = extract_window_patches_gpu(img_n, i, j, f, t, m, n)

    # Grafo kNN com pesos = distâncias
    edges_gdf = knn_graph_edges_gpu(dataset_cp, nn)

    # SSSP a partir do nó central
    df_sssp = sssp_gpu(edges_gdf, source_idx)  # cudf: vertex, distance, predecessor

    # Similaridade: exp(-d^2 / h^2)
    # Converte para CuPy para operar no device
    vertices_cp = df_sssp['vertex'].values  # device array
    dist_cp = df_sssp['distance'].values.fillna(cp.inf)

    # Alguns nós podem ficar com +inf (desconexos) -> zera contribuição
    valid_mask = cp.isfinite(dist_cp)
    dist_valid = dist_cp[valid_mask]
    vert_valid = vertices_cp[valid_mask]

    sims = cp.exp(- (dist_valid ** 2) / (h ** 2))
    pix = pixels_busca_cp[vert_valid]

    NL = (sims * pix).sum()
    Z = sims.sum()

    # Evita divisão por zero
    val = (NL / Z) if Z > 0 else img_n[i + f, j + f]
    return float(cp.asnumpy(val))


def geonlm_gpu(ruidosa_np: np.ndarray, f=4, t=7, h_geo=150.0, nn=10) -> np.ndarray:
    """
    Versão principal GEONLM: recebe imagem ruidosa (NumPy), processa em GPU e devolve NumPy.
    """
    # Sobe imagem para GPU
    ruidosa_cp = cp.asarray(ruidosa_np, dtype=cp.float32)

    m, n = ruidosa_cp.shape
    img_n = pad_symmetric_gpu(ruidosa_cp, f)

    filtrada = np.empty((m, n), dtype=np.float32)

    # Loop na CPU chamando kernels GPU internamente (kNN/SSSP)
    # Dica: experimente blocar por linhas/colunas para melhor ocupação (futuro).
    for i in range(m):
        for j in range(n):
            filtrada[i, j] = geonlm_pixel_gpu(img_n, i, j, f, t, h_geo, nn, m, n)

    return filtrada


# -----------------------------
# Demo simples (equivalente ao seu main)
# -----------------------------

if __name__ == "__main__":
    # Carrega imagem
    img = skimage.io.imread('ProjetoDoutorado/wvc/images/0.gif')
    img = img[0, :, :] if len(img.shape) > 2 else img
    if img.ndim > 2:
        img = skimage.color.rgb2gray(img) * 255.0

    img = img.astype(np.float32)
    m, n = img.shape

    print(f'Num. linhas = {m}')
    print(f'Num. colunas = {n}\n')

    # Adiciona ruído gaussiano
    sigma = 10.0
    ruido = np.random.normal(0.0, sigma, (m, n)).astype(np.float32)
    ruidosa = img + ruido
    ruidosa = np.clip(ruidosa, 0.0, 255.0)

    # Parâmetros
    f = 4
    t = 7
    h_geo = 150.0
    nn = 10

    print('***********************************')
    print('*            GEONLM CUDA          *')
    print('***********************************\n')

    t0 = time.time()
    filtrada_geo = geonlm_gpu(ruidosa, f=f, t=t, h_geo=h_geo, nn=nn)
    t1 = time.time()

    # Métricas (CPU)
    psnr_geo = peak_signal_noise_ratio(img.astype(np.uint8), filtrada_geo.clip(0,255).astype(np.uint8))
    ssim_geo = structural_similarity(img.astype(np.uint8), filtrada_geo.clip(0,255).astype(np.uint8))

    print(f'\nPSNR (GEO NLM): {psnr_geo:.4f}')
    print(f'SSIM (GEO NLM): {ssim_geo:.4f}')
    print(f'Tempo total (GEONLM CUDA): {t1 - t0:.3f} s\n')

    skimage.io.imsave('GEONLM_cuda.png', filtrada_geo.clip(0,255).astype(np.uint8))
