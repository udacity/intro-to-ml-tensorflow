# If you are unable to use the GPU inside the container, see the solution
# in the following link.
# https://forums.developer.nvidia.com/t/nvida-container-toolkit-failed-to-initialize-nvml-unknown-error/286219
FROM tensorflow/tensorflow:2.16.1-gpu-jupyter

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update \
 && export DEBIAN_FRONTEND=noninteractive \
 && apt-get -y install --no-install-recommends \
    neovim \
 && apt-get clean autoclean \
 && apt-get autoremove --yes 
# && rm -rf /var/lib/apt/lists/*
# && rm -rf /var/lib/{apt,dpkg,cache,log}/

COPY requirements.txt /tmp/pip-tmp/

RUN pip install --upgrade pip \
 && pip install --upgrade --upgrade-strategy only-if-needed --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
 && rm -rf /tmp/pip-tmp

RUN echo "alias vi=nvim" >> ~/.bash_aliases \
 && echo "alias vim=nvim" >> ~/.bash_aliases
 
