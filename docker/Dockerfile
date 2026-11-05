# The base must already supply the vendor torch/extension and Triton/FlagTree.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
WORKDIR /opt/lightx2v-app
# Install ffmpeg/libav and the vendor-matched torchvision with the base image's
# package manager. These are checked below instead of replacing vendor packages.
RUN ffmpeg -version && ffprobe -version
COPY docker/requirements-h3.txt /opt/lightx2v-app/requirements-h3.txt
RUN python3 -m pip install --no-cache-dir -r requirements-h3.txt
COPY . /opt/FlagGems-lightx2v
RUN python3 -m pip install --no-deps /opt/FlagGems-lightx2v
# FlagGems, LightX2V/lightx2v_platform and the plugin must be installed at their
# reviewed commits in a site-specific child image. No model weights are copied.
CMD ["python3", "/opt/FlagGems-lightx2v/examples/operator_smoke.py", "--help"]
