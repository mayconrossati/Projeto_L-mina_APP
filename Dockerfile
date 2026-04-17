# 1. Usa uma imagem oficial do Python super leve (slim) para o servidor
FROM python:3.12-slim

# 2. Define a pasta principal lá dentro do servidor
WORKDIR /app

# 3. Copia apenas a lista de dependências primeiro (Melhora a velocidade do Docker)
COPY requirements.txt .

# 4. Instala todas as bibliotecas (IA, Web3, Streamlit, etc.) sem guardar lixo de cache
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia o seu código Python para dentro do servidor
COPY lumina_app.py .

# 6. Libera a porta 8501 (que é a porta oficial do Streamlit)
EXPOSE 8501

# 7. O comando exato que o servidor vai rodar quando ligar a máquina
CMD ["streamlit", "run", "lumina_app.py", "--server.port=8501", "--server.address=0.0.0.0"]