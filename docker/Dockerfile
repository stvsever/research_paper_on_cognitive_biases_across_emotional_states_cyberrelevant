FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
	&& apt-get install -y --no-install-recommends make \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY config /workspace/config

RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -e .

# Create output directories
RUN mkdir -p /workspace/paper/assets/figures \
	&& mkdir -p /workspace/paper/assets/tables \
	&& mkdir -p /workspace/src/review_stages/analysis_outputs

CMD ["cbias", "run-all"]
