FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgl1 \
  patchelf \
  && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv pip install --system --no-cache -r pyproject.toml

RUN patchelf --clear-execstack /usr/local/lib/python3.11/site-packages/inspireface/modules/core/libs/linux/x64/libInspireFace.so

COPY . .

EXPOSE 8000
