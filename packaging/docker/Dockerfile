# syntax=docker/dockerfile:1
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
COPY --from=ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11vnc openbox xauth x11-utils libglib2.0-0 libgomp1 libegl1 libgl1 \
    libdbus-1-3 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0 libxcb-render-util0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
ARG ACCELERATOR=cpu
RUN test "$ACCELERATOR" = cpu || test "$ACCELERATOR" = cuda
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable --no-install-project --extra "$ACCELERATOR" --extra qt --extra desktop
COPY src ./src
ARG SOURCE_REVISION
ARG SOURCE_DIRTY=true
RUN SOURCE_REVISION="$SOURCE_REVISION" SOURCE_DIRTY="$SOURCE_DIRTY" python -c \
    'import json,os,pathlib; r=os.environ["SOURCE_REVISION"]; assert len(r)==40 and all(c in "0123456789abcdef" for c in r); pathlib.Path("src/organoidtracker/_build_info.json").write_text(json.dumps({"source_revision":r,"dirty":os.environ["SOURCE_DIRTY"]!="false"}))'
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable --extra "$ACCELERATOR" --extra qt --extra desktop
ADD --checksum=sha256:b1003a11b6e6e8d8f7f5e5586daae7f8ca651d8aee0aa155ff9ac841c48f52c6 https://codeload.github.com/novnc/noVNC/tar.gz/refs/tags/v1.7.0 /tmp/novnc.tar.gz
RUN mkdir /opt/novnc && tar -xzf /tmp/novnc.tar.gz --strip-components=1 -C /opt/novnc && rm /tmp/novnc.tar.gz
COPY packaging/docker/prefer-lossless.py /opt/prefer-lossless.py
RUN python /opt/prefer-lossless.py
COPY packaging/docker/index.html /opt/novnc/index.html
COPY packaging/licenses /opt/licenses
COPY LICENSE NOTICE /opt/licenses/organoidtracker/
# Retain matching source and installed license metadata in the image. No weights,
# tests, private input or host preferences are part of the build context.
RUN /opt/venv/bin/python -c 'import importlib.metadata as m,json,pathlib; pathlib.Path("/opt/licenses/packages.json").write_text(json.dumps(sorted([{"name":d.metadata["Name"],"version":d.version} for d in m.distributions()],key=lambda p:p["name"].lower()),indent=2))'
ENV PATH="/opt/venv/bin:$PATH" DISPLAY=:99 QT_QPA_PLATFORM=xcb \
    HOME=/tmp/home XDG_CACHE_HOME=/tmp/cache XDG_CONFIG_HOME=/tmp/config \
    MPLCONFIGDIR=/tmp/matplotlib ORGANOIDTRACKER_CHECKPOINTS=/models \
    OT_DEVICE=$ACCELERATOR
WORKDIR /tmp
USER 1000:1000
STOPSIGNAL SIGTERM
EXPOSE 6080
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=3 CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:6080/api/status'))['ready']"
ENTRYPOINT ["python", "-m", "organoidtracker.desktop.supervisor"]
