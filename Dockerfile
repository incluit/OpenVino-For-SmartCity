FROM ubuntu:16.04

ADD . /app
WORKDIR /app

ARG INSTALL_DIR=/opt/intel/computer_vision_sdk

RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Pick up some TF dependencies
RUN apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3-pip \
        cmake \
        sudo 

RUN pip3 install --upgrade pip setuptools wheel

# installing OpenVINO dependencies
RUN cd /app/l_openvino_toolkit* && \
    ./install_cv_sdk_dependencies.sh

## installing OpenVINO itself
RUN cd /app/l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites/ && \
    ./install_prerequisites_tf.sh

RUN /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh"

RUN cd /app && \
    rm -rf l_openvino_toolkit*

CMD ["/bin/bash"]
