# Base image
FROM python:3.7-slim

# Install wget
FROM ubuntu:20.04
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*


COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY config/ config/

# Path configuration
# ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Installs google cloud sdk, this is mostly for using gsutil to export model.
#RUN wget -nv \
#    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
#    mkdir /root/tools && \
#    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
#    rm google-cloud-sdk.tar.gz && \
#    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
#        --path-update=false --bash-completion=false \
#        --disable-installation-options && \
#    rm -rf /root/.config/* && \
#    ln -s /root/.config /config && \
#    # Remove the backup directory that gcloud creates
#    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# WORKDIR /
#RUN set -xe \
#    && apt-get update \
#    && apt-get install python3-pip
#RUN pip install --upgrade pip


# Pip stuff
RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip

RUN pip install --upgrade pip

RUN pip install -r requirements.txt --no-cache-dir

ARG wand_api
ENV env_wand_api=wand_api

ENTRYPOINT ["python3", "-u", "src/models/train_model.py"]