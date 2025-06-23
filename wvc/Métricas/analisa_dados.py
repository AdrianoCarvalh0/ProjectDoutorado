import numpy as np
import pandas
from scipy.stats import wilcoxon, friedmanchisquare
import scikit_posthocs as sp

############ Gaussian noise
print('Gaussian noise')
print('----------------')
print()
df = pandas.read_csv('metricas_gaussian.csv')
# Converte para matriz (numpy)
dados = df.to_numpy()

# Substitui vírgulas por pontos 
for i in range(dados.shape[0]):
    for j in range(dados.shape[1]):
        dados[i, j] = float(dados[i, j].replace(',', '.'))

# Separa os grupos
ssim_nlm = dados[:, 0].astype(float)
ssim_geo = dados[:, 1].astype(float)
ssim_bm3d = dados[:, 2].astype(float)
psnr_nlm = dados[:, 3].astype(float)
psnr_geo = dados[:, 4].astype(float)
psnr_bm3d = dados[:, 5].astype(float)

# Realiza o teste de Friedman para SSIM's
print('SSIM')
print('Mediana NLM:', np.median(ssim_nlm))
print('Mediana GEONLM:', np.median(ssim_geo))
print('Mediana BM3D:', np.median(ssim_bm3d))
print()

print(friedmanchisquare(ssim_nlm, ssim_geo, ssim_bm3d))
print(sp.posthoc_nemenyi_friedman(dados[:, :3]))
print()

print('*** Do ponto de vista de SSIM, GEONLM é estatisticamente superior ao NLM padrão.')
print('*** Do ponto de vista de SSIM, BM3D é estatisticamente superior ao GEONLM.')
print()

# Realiza o teste de Friedman para PSNR's
print('PSNR')
print('Mediana NLM:', np.median(psnr_nlm))
print('Mediana GEONLM:', np.median(psnr_geo))
print('Mediana BM3D:', np.median(psnr_bm3d))
print()

print('*** Do ponto de vista de PSNR, GEONLM é estatisticamente superior ao NLM padrão.')
print('*** Do ponto de vista de PSNR, BM3D é equivalente ao GEONLM.')
print()

print(friedmanchisquare(psnr_nlm, psnr_geo, psnr_bm3d))
print(sp.posthoc_nemenyi_friedman(dados[:, 3:]))
print()
input('Pressione enter para continuar...')
print()

############ Salt and pepper noise
print('Salt and pepper noise')
print('-----------------------')
print()
df = pandas.read_csv('metricas_saltpepper.csv')
# Converte para matriz (numpy)
dados = df.to_numpy()

# Substitui vírgulas por pontos 
for i in range(dados.shape[0]):
    for j in range(dados.shape[1]):
        dados[i, j] = float(dados[i, j].replace(',', '.'))

# Separa os grupos
ssim_nlm = dados[:, 0].astype(float)
ssim_geo = dados[:, 1].astype(float)
ssim_bm3d = dados[:, 2].astype(float)
ssim_med = dados[:, 3].astype(float)
psnr_nlm = dados[:, 4].astype(float)
psnr_geo = dados[:, 5].astype(float)
psnr_bm3d = dados[:, 6].astype(float)
psnr_med = dados[:, 7].astype(float)

# Realiza o teste de Friedman para SSIM's
print('SSIM')
print('Mediana NLM:', np.median(ssim_nlm))
print('Mediana GEONLM:', np.median(ssim_geo))
print('Mediana BM3D:', np.median(ssim_bm3d))
print('Mediana MEDIANA:', np.median(ssim_med))
print()

print('*** Do ponto de vista de SSIM, GEONLM é estatisticamente superior ao NLM padrão e ao BM3D.')
print('*** Do ponto de vista de SSIM, filtro da mediana é equivalente ao GEONLM.')
print()

print(friedmanchisquare(ssim_nlm, ssim_geo, ssim_bm3d, ssim_med))
print(sp.posthoc_nemenyi_friedman(dados[:, :4]))
print()

# Realiza o teste de Friedman para PSNR's
print('PSNR')
print('Mediana NLM:', np.median(psnr_nlm))
print('Mediana GEONLM:', np.median(psnr_geo))
print('Mediana BM3D:', np.median(psnr_bm3d))
print('Mediana MEDIANA:', np.median(psnr_med))
print()

print('*** Do ponto de vista de PSNR, GEONLM é estatisticamente superior ao NLM padrão e ao BM3D.')
print('*** Do ponto de vista de PSNR, filtro da mediana é equivalente ao GEONLM.')
print()

print(friedmanchisquare(psnr_nlm, psnr_geo, psnr_bm3d, psnr_med))
print(sp.posthoc_nemenyi_friedman(dados[:, 4:]))
