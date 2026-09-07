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
