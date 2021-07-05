FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
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
     software-properties-common
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install setuptools==45.2.0
RUN pip3 install scipy==1.5.4
RUN pip3 install numpy==1.18.5
RUN pip3 install argparse scikit-learn tensorflow-gpu==2.3.1 Pillow pandas

