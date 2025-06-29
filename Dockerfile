# Usa una imagen base de Python. Python 3.11 es muy estable y compatible.
# Evitamos 3.12 y 3.13 por ahora para maximizar la compatibilidad de wheels con numpy 2.x y scikit-learn.
# Vamos a intentar Python 3.11.9, que ya sabemos que Render soporta.
FROM python:3.11.9-slim-bullseye

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo requirements.txt primero para aprovechar el cache de Docker
# Esto instala las dependencias una sola vez, a menos que requirements.txt cambie
COPY requirements.txt .

# Instala las dependencias de Python
# `--no-cache-dir` para evitar almacenar cache de pip y reducir tamaño de imagen
# `--upgrade pip` para asegurar que pip esté actualizado y resuelva mejor
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de tu aplicación
# Esto copiará app.py, preprocessing.py, la carpeta models/, static/, templates/, etc.
COPY . .

# Expone el puerto en el que Gunicorn escuchará. Render inyecta esto automáticamente,
# pero es una buena práctica incluirlo. Gunicorn usará la variable de entorno PORT.
EXPOSE 10000

# Comando para iniciar la aplicación con Gunicorn
# Usa la variable de entorno $PORT que Render inyecta
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]
