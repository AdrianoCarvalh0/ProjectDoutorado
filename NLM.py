import warnings
import numpy as np
from tqdm import tqdm

# Para evitar warning de divisão por zero
warnings.simplefilter(action='ignore')

'''
Non-Local Means padrão

Parâmetros:

    img: imagem ruidosa de entrada
    h: parâmetro que controla o grau de suavização (quanto maior, mais suaviza)
    f: tamanho do patch (2f + 1 x 2f + 1) -> se f = 3, então patch é 7 x 7
    t: tamanho da janela de busca (2t + 1 x 2t + 1) -> se t = 10, então janela de busca é 21 x 21

'''
def NLM(img, h, f, t):

    
    # Dimenssões espaciais da imagem
    m, n = img.shape

    # Cria imagem de saída
    filtrada = np.zeros((m, n))

    # Problema de valor de contorno: replicar bordas
    #img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric') # Modifiquei esta linha
    img_n = np.pad(img, ((f, f), (f, f)), 'symmetric')


    # Initializing the counter
    prog = tqdm(total=m*n, position=0, leave=True)

    # Loop principal do NLM
    for i in range(m):
        for j in range(n):

            im = i + f;   # compensar a borda adicionada artificialmente
            jn = j + f;   # compensar a borda adicionada artificialmente

            # Obtém o patch ao redor do pixel corrente
            W1 = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]

            # Calcula as bordas da janela de busca para o pixel corrente (se pixel próximo das bordas, janela de busca é menor)
            rmin = max(im-t, f);  # linha inicial
            rmax = min(im+t, m+f);  # linha final
            smin = max(jn-t, f);  # coluna inicial
            smax = min(jn+t, n+f);  # coluna final

            # Calcula média ponderada
            NL = 0      # valor do pixel corrente filtrado
            Z = 0       # constante normalizadora

            # Loop para todos os pixels da janela de busca
            for r in range(rmin, rmax):
                for s in range(smin, smax):

                    # Obtém o patch ao redor do pixel a ser comparado
                    W2 = img_n[r-f:(r+f)+1, s-f:(s+f)+1]

                    # Calcula o quadrado da distância Euclidiana
                    d2 = np.sum((W1 - W2)*(W1 - W2))

                    # Calcula a medida de similaridade
                    sij = np.exp(-d2/(h**2))

                    # Atualiza Z e NL
                    Z = Z + sij
                    NL = NL + sij*img_n[r, s]

            # Normalização do pixel filtrado
            filtrada[i, j] = NL/Z
            prog.update(1)
    return filtrada

    
