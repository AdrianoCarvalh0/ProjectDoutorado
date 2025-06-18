#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:26:44 2019

Implementação do filtro Non-Local Means geodésico

"""
import cupy as cp
import sys
import warnings
import time
import skimage
import statistics
import networkx as nx
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import sklearn.neighbors as sknn
from scipy.sparse.csgraph import dijkstra
from numpy.matlib import repmat
from scipy.linalg import eigh
from numpy.linalg import inv
from numpy.linalg import cond
from numpy import eye
from sklearn.decomposition import PCA
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from numba import njit   # just in time compiler (acelera loops)
from joblib import Parallel, delayed

# Para evitar warning de divisão por zero
warnings.simplefilter(action='ignore')

'''
Espelhamento das bordas da imagem A de maneira simétrica
A função pad do numpy não é supertada pelo numba!
Substitui a função: img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')
f é o parâmetro (largura das bordas)
'''


@njit
def mirror(A, f):
    n = A.shape[0]
    m = A.shape[1]
    nlin = A.shape[0] + 2*f
    ncol = A.shape[1] + 2*f
    # Matriz de saída
    B = np.zeros((nlin, ncol))
    # Preeenche miolo
    B[f:nlin-f, f:ncol-f] = A
    # Preenche cantos
    B[0:f, 0:f] = np.flip(A[0:f, 0:f])                          # 1o quadrante
    B[0:f, ncol-f:ncol] = np.flip(A[0:f, m-f:m])                # 2o quadrante
    B[nlin-f:nlin, 0:f] = np.flip(A[n-f:n, 0:f])                # 3o quadrante
    B[nlin-f:nlin, ncol-f:ncol] = np.flip(A[n-f:n, m-f:m])      # 4o quadrante
    # Preenche bordas
    B[0:f, f:ncol-f] = np.flipud(A[0:f, :])             # cima
    B[nlin-f:nlin, f:ncol-f] = np.flipud(A[n-f:n, :])   # baixo
    B[f:nlin-f, 0:f] = np.fliplr(A[:, 0:f])             # esquerda
    B[f:nlin-f, ncol-f:ncol] = np.fliplr(A[:, m-f:m])   # direita
    return B

######################################################
# Função auxiliar para paralelizar o GEONLM
######################################################
def process_pixel(i, j, img_n, f, t, h, nn):
    im = i + f
    jn = j + f
    patch_central = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]
    central = np.reshape(patch_central, [1, patch_central.shape[0]*patch_central.shape[1]])[-1]
    rmin = max(im-t, f)
    rmax = min(im+t, m+f)
    smin = max(jn-t, f)
    smax = min(jn+t, n+f)
    NL, Z = 0, 0
    dataset = np.zeros(((rmax - rmin)*(smax - smin), (2*f + 1)*(2*f + 1)))
    k = 0
    pixels_busca = []
    for r in range(rmin, rmax):
        for s in range(smin, smax):
            W = img_n[r-f:(r+f)+1, s-f:(s+f)+1]
            neighbor = np.reshape(W, [1, W.shape[0]*W.shape[1]])[-1]
            dataset[k, :] = neighbor.copy()
            if central[0] == neighbor[0] and (central == neighbor).all():
                source = k
            pixels_busca.append(img_n[r, s])
            k += 1
    knnGraph = sknn.kneighbors_graph(dataset, n_neighbors=nn, mode='distance')
    A = knnGraph.toarray()
    G = nx.from_numpy_array(A)
    length, path = nx.single_source_dijkstra(G, source)
    points = np.array(list(length.keys()))
    distancias = np.array(list(length.values()))
    similaridades = np.exp(-distancias**2 / (h**2))
    pixels_busca = np.array(pixels_busca)
    pixels = pixels_busca[points]
    NL = sum(similaridades * pixels)
    Z = sum(similaridades)
    return NL / Z

##################################################
# GEONLM paralelo 
##################################################
def Parallel_GEONLM(img_n, f, t, h, nn):
    # Parallelize the loop
    m = img_n.shape[0] - 2*f
    n = img_n.shape[1] - 2*f
    filtrada = Parallel(n_jobs=-1)(delayed(process_pixel)(i, j, img_n, f, t, h, nn) for i in range(m) for j in range(n))
    filtrada_geo = np.array(filtrada).reshape((m, n))
    return filtrada_geo

####################################################################
'''
Função que extrai os patches de cada janela de busca no GeoNLM
Retorna uma matriz 4D (m, n, 2t+1 x 2t+1, 2f+1 x 2f+1)

Usa o JIT (just in time) compiler para acelerar loops

'''
####################################################################
@njit
def Extract_patches(img, f, t):
    # Dimenssões espaciais da imagem
    m, n = img.shape
    # Tamanhos do patch e da janela de busca
    tamanho_patch = (2*f + 1)*(2*f + 1)    
    # Patches para cada janela de busca
    patches = []
    centros = []    
    # Problema de valor de contorno: replicar bordas
    img_n = mirror(img, f)
    # Loop principal do NLM geodésico
    for i in range(m):        
        for j in range(n):
            im = i + f;   # compensar a borda adicionada artificialmente
            jn = j + f;   # compensar a borda adicionada artificialmente
            # Obtém o patch ao redor do pixel corrente
            patch_central = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1].copy()
            central = patch_central.reshape((1, patch_central.shape[0]*patch_central.shape[1]))[-1]
            # Calcula as bordas da janela de busca para o pixel corrente
            rmin = max(im-t, f);  # linha inicial
            rmax = min(im+t, m+f);  # linha final
            smin = max(jn-t, f);  # coluna inicial
            smax = min(jn+t, n+f);  # coluna final
            num_elem = (rmax - rmin)*(smax - smin)
            # Cria dataset
            dataset = np.zeros((num_elem, tamanho_patch))
            # Loop para montar o dataset com todos os patches da janela
            k = 0
            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    # Obtém o patch ao redor do pixel a ser comparado
                    W = img_n[r-f:(r+f)+1, s-f:(s+f)+1].copy() 
                    neighbor = W.reshape((1, W.shape[0]*W.shape[1]))[-1]
                    dataset[k, :] = neighbor.copy()
                    if (central == neighbor).all():
                        source = k
                    k = k + 1
            patches.append(dataset)
            centros.append(source)
    return patches, centros




import skimage.io
import skimage.color
import cupy as cp
import numpy as np

img = skimage.io.imread('extras/images/ct2.png')

# Se for colorida, converte para escala de cinza
if len(img.shape) > 2:
    img = skimage.color.rgb2gray(img)  # valores de 0 a 1
    img = 255 * img

img = img.astype(np.uint8)         # ainda em NumPy
img_gpu = np.array(img)            # agora sim, em GPU com cupy

m, n = img_gpu.shape

print('Num. linhas = %d' % m)
print('Num. colunas = %d' % n)
print()

# Ruído
sigma = 10
ruido = np.random.normal(0, sigma, (m, n))

# Imagem ruidosa
ruidosa = img_gpu + ruido

# Clipa imagem para intervalo [0, 255]
ruidosa[np.where(ruidosa > 255)] = 255
ruidosa[np.where(ruidosa < 0)] = 0
#ruidosa = ruidosa.astype(np.float32)

# Listas para métricas
psnrs_nlm = []
ssims_nlm = []

psnrs_geonlm = []
ssims_geonlm = []

# Define parâmetros do filtro NLM
f = 4   # tamanho do patch (2f + 1 x 2f + 1) -> 5 x 5
t = 7   # tamanho da janela de busca (2t + 1 x 2t + 1) -> 21 x 21
h_nlm = 110 # parâmetro que controla a suavização no NLM (depende da imagem)
h_geo = 150 # parâmetro que controla a suavização no GEONLM (depende da imagem)
nn = 10     # número de vizinhos no grafo k-NN

# Cria imagem de saída
filtrada = np.zeros((m, n))

# Problema de valor de contorno: replicar bordas
img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')

####################################################
# Filtra com NLM geodésico
###################################################
print('***********************************')
print('*            GEONLM               *')
print('***********************************')
print()
ini = time.time()

##3 Versão com JIT compiler
#filtrada_geo = GeoNLM_filter(ruidosa, h_geo, f, t, nn)   # versão com JIT compiler (não ficou mais rápido pois o Dijkstra não acelera. Ainda está lento...)

### Versão com paralelismo (ficou mais rápida)
img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')      # para versão paralela, precisa espelhar imagem fora da função
filtrada_geo = Parallel_GEONLM(img_n, f, t, h_geo, nn)

end = time.time()
# Calcula PSNR
p2 = peak_signal_noise_ratio(img, filtrada_geo.astype(np.uint8))
psnrs_nlm.append(p2)
print('\nPSNR (GEO NLM): %f' %p2)
# Calcula SSIM
s2 = structural_similarity(img, filtrada_geo.astype(np.uint8))
ssims_nlm.append(s2)
print('SSIM (GEO NLM): %f' %s2)
print('Total Elapsed time (GEONLM): %f s' %(end - ini))
print()
# Salva arquivo    
skimage.io.imsave('GEONLM.png', filtrada_geo.astype(np.uint8))

