FROM iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:2.3.2

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

# Standalone usd-core (a pxr independent of Kit's): used by inspect_usd.py to inspect uploaded
# USD files. Installed into its own directory so it does not pollute the training environment.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --target /opt/usd-core usd-core
