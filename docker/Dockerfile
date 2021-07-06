FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends gfortran \
     liblapack-dev \
     liblapack3 \
     libopenblas-base \
     libopenblas-dev \
     xorg \
     libx11-dev \
     libglu1-mesa-dev \
     libfreetype6-dev \
     software-properties-common \
     curl
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py
#RUN apt-get update && apt-get install -y python3-pip
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install setuptools==45.2.0
RUN pip install deep_density_estimation
RUN pip install tensorflow-gpu
