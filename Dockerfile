# Same base image as the organizers' evaluation container (Python 3.11, torch 2.4.1, torchvision).
# NOTE: this Dockerfile has not been built by the author (no Docker daemon was available); the same code was run directly on
# Python 3.10 with torch 2.12 and torchvision 0.27 (see README, "What was and was not tested").
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Default command: the offline self-test (synthetic data, no downloads except torchvision's untrained stubs).
CMD ["python", "tools/upar/selftest.py"]
