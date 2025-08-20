# Use a imagem oficial RAPIDS base
FROM rapidsai/base:24.06-cuda12.2-py3.10

# Opcional: Atualiza e instala pacotes úteis
RUN apt-get update && apt-get install -y git vim && rm -rf /var/lib/apt/lists/*

# Dá permissão total na pasta workspace para qualquer usuário
RUN chmod -R a+rw /workspace
