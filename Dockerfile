# poly-stable-diffusion-webui: web UI + HTTP API on an NVIDIA GPU.
#
# The PyTorch CUDA wheels bundle the CUDA runtime, so a slim Python base is enough. The host
# only needs the NVIDIA driver and the NVIDIA Container Toolkit. Model weights are not baked in:
# the SD 1.5 checkpoint downloads on first start if no checkpoint is present, and everything the
# webui writes (models, outputs, extensions, settings) lives under /data (mount volumes there).
#
#   docker build -t poly-stable-diffusion-webui .
#   docker run --gpus all -p 7860:7860 -v sd-data:/data -v sd-models:/data/models -v sd-outputs:/data/outputs poly-stable-diffusion-webui

# The webui is tested on Python 3.10 and warns on anything else.
ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim-bookworm

# The webui's default torch build. CUDA 12.1 wheels run on driver >= 530 and support GPUs from
# Maxwell to Ada. For Blackwell (RTX 50xx) build with a CUDA 12.8 torch instead, e.g.
#   --build-arg TORCH_COMMAND="pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_COMMAND="pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url ${TORCH_INDEX_URL}"

# git: launch.py clones its repositories, and extensions are git repositories.
# libgl1, libglib2.0-0: OpenCV, pulled in by facexlib and many extensions.
# libtcmalloc-minimal4: lower CPU memory use, as webui.sh does with LD_PRELOAD.
# build-essential: extensions whose install.py builds a native package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 libtcmalloc-minimal4 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Everything the webui installs is owned by the app user, so installing extensions (and their
# requirements) from the UI keeps working at runtime.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /opt/venv /app /data \
    && chown app:app /opt/venv /app /data

USER app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch first, in its own layer: it is by far the largest and changes least. launch.py runs the
# same command (with "python -m") when torch is missing.
RUN python -m ${TORCH_COMMAND}

# Then the pinned requirements, so source changes don't reinstall them.
#
# CLIP is installed here rather than by launch.py: its setup.py imports pkg_resources, which
# current setuptools no longer ships, so pip's isolated build fails. Building it against the
# pinned setuptools from requirements_versions.txt (plus wheel) works. Same URL as CLIP_PACKAGE
# in modules/launch_utils.py.
ARG CLIP_PACKAGE=https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip
WORKDIR /app
COPY --chown=app:app requirements_versions.txt ./
RUN pip install -r requirements_versions.txt wheel \
    && pip install --no-build-isolation ${CLIP_PACKAGE}

# onnxruntime for the extensions that use it (ReActor, ControlNet's DWPose, Rembg). ReActor
# requires exactly this CUDA 12 build, from Microsoft's feed, whenever it sees a GPU.
ARG ONNXRUNTIME_PACKAGE=onnxruntime-gpu==1.17.1
ARG ONNXRUNTIME_INDEX_URL=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# launch.py does the rest of the setup: CLIP, open_clip, the repositories it clones, and the
# extensions in bundled-extensions.txt with the packages their install.py scripts install. It
# runs again on every start, finds everything in place and moves on.
#
# The pip cache mount only speeds up rebuilds; it is not part of the image. There is no GPU
# while building, so the extensions install the CPU onnxruntime and ReActor records "CPU" as
# its device; both are put right afterwards. ReActor's face-swap model is a download like any
# other model weights, so it is removed here and fetched into the models volume on first start.
COPY --chown=app:app . .
RUN --mount=type=cache,target=/home/app/.cache/pip,uid=1000,gid=1000 \
    env -u PIP_NO_CACHE_DIR python launch.py --skip-torch-cuda-test --exit \
    && pip uninstall -y onnxruntime onnxruntime-gpu \
    && env -u PIP_NO_CACHE_DIR pip install ${ONNXRUNTIME_PACKAGE} --extra-index-url ${ONNXRUNTIME_INDEX_URL} \
    && rm -f extensions/*/last_device.txt \
    && rm -rf models/insightface \
    && find /opt/venv /app -name "__pycache__" -type d -prune -exec rm -rf {} +

# Seed /data with the repository's placeholders, small bundled models (VAE-approx, karlo) and
# the bundled extensions. A new, empty named volume is initialised from this. A data volume
# from an older image gets missing bundled extensions copied from /app/extensions on start.
RUN mkdir -p /data/models/sam /data/models/ControlNet /data/embeddings /data/extensions /data/outputs /data/cache \
    && cp -a models/. /data/models/ \
    && cp -a embeddings/. /data/embeddings/ \
    && cp -a extensions/. /data/extensions/

ENV PYTHONUNBUFFERED=1 \
    LD_PRELOAD=libtcmalloc_minimal.so.4 \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SD_WEBUI_RESTART=tmp/restart \
    COMMANDLINE_ARGS=""

WORKDIR /app
VOLUME ["/data", "/data/models", "/data/outputs"]
EXPOSE 7860

# The first start downloads the SD 1.5 checkpoint (~4 GB) before the server listens, and with
# a venv volume from an older image, installs the extensions' packages too.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1200s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/internal/ping', timeout=4)"

# Extra webui flags (e.g. "--xformers --medvram") go in COMMANDLINE_ARGS, which launch.py appends
# to these. SD_WEBUI_RESTART enables the UI's restart button: the process exits and the container
# restart policy starts it again.
CMD ["python", "launch.py", "--listen", "--port", "7860", "--api", "--data-dir", "/data"]
