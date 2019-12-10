FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG UID
ARG GID
ARG USERNAME

ENV UID=${UID} GID=${GID} USERNAME=${USERNAME}

########## ↓↓↓ root ↓↓↓ ##########
# ユーザ設定
RUN apt-get update && apt-get install -y sudo libxrender1 libxext6 && \
    groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -G sudo -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME}

# 言語設定
RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc \
    && echo "export LANG=ja_JP.UTF-8" >> /home/${USERNAME}/.bashrc

# パッケージインストール
RUN apt-get -qq update \
    && apt-get -qq -y install curl bzip2 libmysqlclient-dev gcc git openssh-server vim

    
########## ↓↓↓ user ↓↓↓ ##########
USER ${USERNAME}
SHELL [ "/bin/bash", "-c" ]

WORKDIR /home/$USERNAME/

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /home/$USERNAME/miniconda3 \
    && rm -rf /tmp/miniconda.sh

ENV PATH /home/$USERNAME/miniconda3/bin:$PATH

RUN conda init bash
RUN conda update conda \
    && sudo apt-get -qq -y remove curl bzip2 \
    && sudo apt-get -qq -y autoremove \
    && sudo apt-get autoclean \
    && sudo rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

COPY ./dti.yml ./dti.yml
RUN conda env create --file dti.yml

