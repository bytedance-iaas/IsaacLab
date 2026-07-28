FROM iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:2.3.2

COPY . /workspace/isaaclab/

RUN git config --system url."https://ghproxy.cc/github.com/".insteadOf "https://github.com/" \
      && git config --system http.sslVerify false

RUN cd /workspace/isaaclab && ./isaaclab.sh --install

RUN cd /workspace/isaaclab/soarm101 && \
      /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps
