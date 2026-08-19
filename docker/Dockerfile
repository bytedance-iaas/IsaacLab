FROM iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:2.3.2

# Package sources. The mirrors below live on the Volcengine network and are unreachable from a
# public runner, so every mirror-dependent step is gated on this flag. Build with
# --build-arg NO_MIRROR=1 to go straight to the upstream sources instead.
#
# Default is mirror mode: this image is built for, and deployed on, Volcengine.
ARG NO_MIRROR
ENV NO_MIRROR=${NO_MIRROR}

# Base image is Ubuntu 22.04 (jammy). Both source layouts are handled because the format
# changed in 24.04: deb822 (.sources) from 24.04 on, the one-line sources.list before it.
# Whichever this image happens to carry gets rewritten; the other branch is a no-op.
RUN if [ "$NO_MIRROR" != "1" ]; then \
        if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
            sed -i \
                -e 's@//.*archive.ubuntu.com@//mirrors.ivolces.com@g' \
                -e 's@//security.ubuntu.com@//mirrors.ivolces.com@g' \
                -e 's@//ports.ubuntu.com@//mirrors.ivolces.com@g' \
                /etc/apt/sources.list.d/ubuntu.sources; \
        fi; \
        if [ -f /etc/apt/sources.list ]; then \
            sed -i \
                -e 's@//.*archive.ubuntu.com@//mirrors.ivolces.com@g' \
                -e 's@//security.ubuntu.com@//mirrors.ivolces.com@g' \
                -e 's@//ports.ubuntu.com@//mirrors.ivolces.com@g' \
                /etc/apt/sources.list; \
        fi; \
    fi

# PyPI stays on Aliyun rather than Volcengine: the Volcengine PyPI mirror only carries upstream
# PyPI projects, and packages published outside it resolve to 404 there.
#
# Set as ENV, not just `pip config`, so it survives into the running container -- anyone who
# pip-installs inside the pod gets the mirror too.
ENV PIP_INDEX_URL=${NO_MIRROR:+"https://pypi.org/simple/"}
ENV PIP_INDEX_URL=${PIP_INDEX_URL:-"https://mirrors.aliyun.com/pypi/simple"}
ENV PIP_TRUSTED_HOST=${NO_MIRROR:+"pypi.org"}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-"mirrors.aliyun.com"}

# Hugging Face downloads (datasets, pretrained checkpoints) through the domestic mirror.
ENV HF_ENDPOINT=${NO_MIRROR:+"https://huggingface.co"}
ENV HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

# Written to pip.conf as well. Every pip below runs through isaaclab.sh, which starts the Isaac
# Sim bundled interpreter in a fresh shell; a config file applies no matter how the environment
# is passed down, and it keeps applying if someone clears the variables at runtime.
# Written directly rather than via `pip config set` so this layer does not depend on where the
# interpreter lives in the base image.
RUN mkdir -p /root/.config/pip && \
    printf '[global]\nindex-url = %s\ntrusted-host = %s\n' \
        "${PIP_INDEX_URL}" "${PIP_TRUSTED_HOST}" > /root/.config/pip/pip.conf

COPY . /workspace/isaaclab/

RUN git config --system url."https://ghproxy.cc/github.com/".insteadOf "https://github.com/" \
      && git config --system http.sslVerify false

RUN cd /workspace/isaaclab && ./isaaclab.sh --install

# Overwrite the rsl-rl-lib 5.0.1 that --install pulls in with our port of it.
# The port is the official 5.0.1 source plus the SAC stack (models/, sac.py,
# off_policy_runner.py, replay_buffer.py) and two backward-compatible patches: mlp.py regains
# its layer_norm argument, and logger.py accepts alpha and collection_size_override. The PPO
# path is unchanged from official 5.0.1, so the soarm101 and Franka PPO demos need no
# adjustment; SAC and OffPolicyRunner are added on top.
# --force-reinstall is required because the version number is identical; --no-deps leaves
# torch and the rest of the environment untouched.
RUN cd /workspace/isaaclab/rsl_rl_port && \
      /workspace/isaaclab/isaaclab.sh -p -m pip install --no-deps --force-reinstall .

RUN cd /workspace/isaaclab/soarm101 && \
      /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps

# TOS python SDK: training-service/upload_tos.py uses it to push checkpoints to Volcengine TOS
# once a training job finishes.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install tos

# LiveKit SDK: with --stream, training publishes a panorama and a closeup view to LiveKit
# (training-service/livekit_stream.py). Demo use only, at small num_envs (~16); real training
# runs headless at high env counts without --stream.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install livekit livekit-api

# Standalone usd-core (a pxr independent of Kit's): used by inspect_usd.py to inspect uploaded
# USD files. Installed into its own directory so it does not pollute the training environment.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --target /opt/usd-core usd-core
