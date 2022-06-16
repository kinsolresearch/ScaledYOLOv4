# We use a two-step build process here to reduce image size.  The
# first image builds mish_cuda from source based on nvidia's CUDA dev
# image.  The second copies the newly built mish_cuda into nvidia's
# smaller CUDA runtime image.
#
# Note the pytorch CUDA architecture list is set to support cards with
# compute capability version 6.1, 7.5, and 8.6, which covers
# GeForce GTX 10xx, GTX 1650, RTX 20xx, and RTX 30xx

FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04 as BUILD

ENV MY_ROOT=/workspace \
    PKG_PATH=/yolo_src \
    NUMPROC=4 \
    PYTHON_VER=3.8 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6"

WORKDIR $PKG_PATH

RUN apt-get update && apt-get install -y apt-utils && apt-get -y dist-upgrade && \
    apt-get install -y git libsnappy-dev libopencv-dev libhdf5-serial-dev libboost-all-dev libatlas-base-dev \
        libgflags-dev libgoogle-glog-dev liblmdb-dev curl unzip ninja-build \
        python${PYTHON_VER}-dev && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VER} && \
    # Clean UP
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*  # cleanup to reduce image size

RUN ln -s /usr/bin/python${PYTHON_VER} /usr/bin/python

RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR $MY_ROOT
# We have to install mish-cuda from source due to an issue with one of the header files
ADD https://github.com/thomasbrandon/mish-cuda/archive/master.zip $MY_ROOT/mish-cuda.zip

# If you're running this on a GPU that has compute capability other than
# what is defined above, mish_cuda needs to be built to target your architecture. 
# Add your compute capability to the definition of TORCH_CUDA_ARCH_LIST above,
# you can find it here https://developer.nvidia.com/cuda-gpus
RUN unzip mish-cuda.zip && \
    cd $MY_ROOT/mish-cuda-master && \
    cp external/CUDAApplyUtils.cuh csrc/ && \
    pip install . && \
    cd $PKG_PATH && \
    rm -rf $MY_ROOT/mish-cuda-master


FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 as RUNTIME

ENV MY_ROOT=/workspace \
    PKG_PATH=/yolo_src \
    NUMPROC=4 \
    PYTHON_VER=3.8 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6"

WORKDIR $PKG_PATH

RUN apt-get update && apt-get install -y apt-utils python${PYTHON_VER}-dev && apt-get -y dist-upgrade && \
    apt-get install -y git libsnappy-dev libopencv-dev libhdf5-serial-dev libboost-all-dev libatlas-base-dev \
        libgflags-dev libgoogle-glog-dev liblmdb-dev curl unzip && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VER} && \
    # Clean UP
    apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*  # cleanup to reduce image size

RUN ln -s /usr/bin/python${PYTHON_VER} /usr/bin/python

RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# copy the mish_cuda Python package from the BUILD image into this one.
COPY --from=BUILD /usr/local/lib/python${PYTHON_VER}/dist-packages/mish_cuda /usr/local/lib/python${PYTHON_VER}/dist-packages/mish_cuda

# ADD https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing /weights/yolov4-csp.weights
ADD requirements.txt $PKG_PATH/requirements.txt
RUN pip install --no-cache-dir -r $PKG_PATH/requirements.txt
WORKDIR $PKG_PATH
ADD yolo $PKG_PATH/yolo
ADD train.py $PKG_PATH/train.py
ADD test.py $PKG_PATH/test.py
ADD setup.py $PKG_PATH/setup.py
ADD data $PKG_PATH/data
RUN pip install .

