#!/usr/bin/env bash
set -e
set -x

if ! type source > /dev/null 2>&1; then
    echo "Restarting the script with bash interpreter"
    bash "$0" "$@"
    exit $?
fi

# Install system packages
if [ ! -f done.system.txt ]; then
    apt-get -y update --fix-missing
    apt-get install -y \
        build-essential \
        clang \
        gcc-9 \
        g++-9 \
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
        patchelf
    apt-get clean
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
    update-alternatives --config gcc
    touch done.system.txt
fi

# Download and install conda
if [ ! -f done.conda.txt ]; then
    rm -rf miniconda3 Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
    bash Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -b -p miniconda3
    miniconda3/bin/conda install -q -y -n base --solver classic conda-libmamba-solver
    miniconda3/bin/conda config --set solver libmamba
    miniconda3/bin/conda install -q -y python=3.9
    pip3 install --upgrade pip
    pip3 install plotly  # local visualization in the notebook
    source miniconda3/bin/activate
    pip3 install --upgrade pip
    touch done.conda.txt
fi

# Install Point2CAD dependencies
if [ ! -f done.deps.txt ]; then
    source miniconda3/bin/activate
    pip install \
        numpy==1.23.5 \
        geomdl \
        open3d \
        pyvista \
        rtree \
        scipy \
        torch \
        tqdm \
        trimesh
    export ROOT_PYMESH=$(pwd)/PyMesh
    rm -rf PyMesh
    git clone https://github.com/PyMesh/PyMesh.git
    cd PyMesh
    git checkout 384ba882
    git submodule update --init
    sed -i '43s|cwd="/root/PyMesh/docker/patches"|cwd="'${ROOT_PYMESH}'/docker/patches"|' ${ROOT_PYMESH}/docker/patches/patch_wheel.py
    pip3 install -r ${ROOT_PYMESH}/python/requirements.txt
    ./setup.py bdist_wheel
    rm -rf build_3.9 third_party/build
    python ${ROOT_PYMESH}/docker/patches/patch_wheel.py dist/pymesh2*.whl
    pip3 install dist/pymesh2*.whl
    python -c "import pymesh; pymesh.test()"
    cd ..
    touch done.deps.txt
fi

echo "==============================================================================
Setup complete, now run as follows:
source miniconda3/bin/activate && cd point2cad && python -m point2cad.main"
