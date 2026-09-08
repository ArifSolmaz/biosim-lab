# biosim-lab — main image.
#
# Deliberately does NOT contain OpenFOAM or Elmer: those live in their own
# images (docker/Dockerfile.openfoam, docker/Dockerfile.elmer) so that the core
# stays small and Stages 1-3 remain installable with plain `pip install`.
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="biosim-lab" \
      org.opencontainers.image.description="Open-source virtual laboratory instrument platform" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/biosim-lab/biosim-lab"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYVISTA_OFF_SCREEN=true

# libGL/libGLU and xvfb are needed by VTK/PyVista even for off-screen rendering;
# libgomp is needed by scikit-image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglu1-mesa \
        libxrender1 \
        libxcursor1 \
        libxinerama1 \
        libsm6 \
        libice6 \
        libgomp1 \
        xvfb \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first so the dependency layer is cached across source edits.
COPY pyproject.toml README.md LICENSE ./
COPY biosim_lab/__init__.py ./biosim_lab/
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install kaleido pytest && \
    # Gmsh publishes no wheel for every platform (notably linux/arm64). It is an
    # optional extra precisely so this is a missing capability, not a failed
    # build: core.geometry falls back to a structured mesh and `biosim doctor`
    # reports it. Try it, and carry on if it is unavailable.
    (pip install "gmsh>=4.12" "meshio>=5.3" \
     || echo "gmsh unavailable on $(uname -m); using the structured-mesh fallback")

# Cellpose, for deep-learning segmentation. It needs PyTorch, and the default
# PyPI wheel carries CUDA on BOTH architectures --- aarch64 included, since Torch
# now ships ARM CUDA builds for GH200-class servers. That is dead weight in a CPU
# image, so both take the CPU-only index, which publishes x86_64 and aarch64
# alike (torch 152 MB against a multi-gigabyte CUDA build). Measured on arm64:
# the CUDA path builds a 12.2 GB image, the CPU index 4.49 GB, against a 3.1 GB
# baseline with no Cellpose at all.
#
# torch and torchvision MUST come from the same index. Mixing them installs
# cleanly and then fails at *import* with "RuntimeError: operator
# torchvision::nms does not exist", so a build that looks fine ships broken.
#
# Optional in the same way gmsh is: a failure here leaves a missing capability
# that `biosim doctor` reports, not a broken build.
RUN ( pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
      && pip install "cellpose>=3.0" ) \
    || echo "cellpose unavailable on $(uname -m); classical segmentation still works"

COPY . .
RUN pip install -e .

# A non-root user, so files written into a mounted volume are not root-owned.
RUN useradd --create-home --uid 1000 biosim && chown -R biosim:biosim /app
USER biosim

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:5006/ >/dev/null || exit 1

# Default: serve the reference scenario's dashboard.
CMD ["biosim", "dashboard", "configs/ctc_vs_rbc.yaml", \
     "--port", "5006", "--address", "0.0.0.0", "--no-show"]
