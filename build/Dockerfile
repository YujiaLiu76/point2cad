FROM nvcr.io/nvidia/pytorch:21.05-py3

RUN apt-get -y update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        gosu \
        gcc \
        g++ \
        git \
        cmake \
        libgmp-dev \
        libmpfr-dev \
        libgmpxx4ldbl \
        libboost-dev \
        libboost-thread-dev \
        libgl1-mesa-glx \
        libxrender1 \
        libspatialindex-dev \
        zip \
        unzip \
        patchelf && \
    apt-get clean

RUN conda install python=3.7 && \
    pip3 install --upgrade pip

ENV ROOT_PROJECT=/work
ENV ROOT_PYMESH ${ROOT_PROJECT}/PyMesh
ENV ROOT_POINT2CAD ${ROOT_PROJECT}/point2cad
RUN mkdir -p ${ROOT_PYMESH} && \
    mkdir -p ${ROOT_POINT2CAD}

WORKDIR ${ROOT_PYMESH}
RUN cd ${ROOT_PYMESH} && \
    git init && \
    git pull https://github.com/PyMesh/PyMesh.git && \
    git checkout 384ba882 && \
    git submodule update --init && \
    sed -i '43s|cwd="/root/PyMesh/docker/patches"|cwd="'${ROOT_PYMESH}'/docker/patches"|' ${ROOT_PYMESH}/docker/patches/patch_wheel.py && \
    pip3 install -r ${ROOT_PYMESH}/python/requirements.txt && \
    ./setup.py bdist_wheel && \
    rm -rf build_3.7 third_party/build && \
    python ${ROOT_PYMESH}/docker/patches/patch_wheel.py dist/pymesh2*.whl && \
    pip3 install dist/pymesh2*.whl && \
    python -c "import pymesh; pymesh.test()"

WORKDIR ${ROOT_PROJECT}
COPY requirements.txt ${ROOT_PROJECT}
RUN cd ${ROOT_PROJECT} && \
    pip install -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR ${ROOT_POINT2CAD}
