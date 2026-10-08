FROM python:3.11-slim-bookworm

# OpenCV needs these system libraries on the slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch. The default Linux wheel bundles CUDA and is several GB larger.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the three YOLO26 weights into the image so suggestions work offline and
# are not downloaded again on every `docker run --rm`.
RUN python -c "from ultralytics import YOLO; [YOLO(w) for w in ('yolo26n.pt', 'yolo26n-seg.pt', 'yolo26n-pose.pt')]"

COPY app.py .

RUN useradd --create-home --uid 1000 app
USER app

# 0.0.0.0 inside the container only. Publish the port on 127.0.0.1 to keep it off your network.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
