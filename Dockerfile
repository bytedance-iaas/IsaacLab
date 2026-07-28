FROM iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:2.3.2

COPY . /workspace/isaaclab/

RUN git config --system url."https://ghproxy.cc/github.com/".insteadOf "https://github.com/" \
      && git config --system http.sslVerify false

RUN cd /workspace/isaaclab && ./isaaclab.sh --install

RUN cd /workspace/isaaclab/soarm101 && \
      /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps

# TOS python SDK: training-service/upload_tos.py uses it to push checkpoints to Volcengine TOS
# once a training job finishes.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install tos

# Standalone usd-core (a pxr independent of Kit's): used by inspect_usd.py to inspect uploaded
# USD files. Installed into its own directory so it does not pollute the training environment.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --target /opt/usd-core usd-core
