FROM --platform=linux/amd64 python:3.10.20-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential latexml && rm -rf /var/lib/apt/lists/*
COPY evaluation/omnidocbench /opt/omnidocbench
RUN python -m pip install --no-cache-dir /opt/omnidocbench
WORKDIR /workspace
# A new build is a new runtime, not the historical image ID. The wrapper records it.
CMD ["python", "pdf_validation.py", "--help"]
