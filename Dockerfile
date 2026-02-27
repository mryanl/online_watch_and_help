FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    unzip \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone -b wah-fix https://github.com/ShunchiZhang/virtualhome.git /workspace/virtualhome \
    && pip install --no-cache-dir -e /workspace/virtualhome

RUN wget -q "http://virtual-home.org/release/simulator/v2.0/v2.2.4/linux_exec.zip" \
    -O /tmp/linux_exec.zip \
    && unzip /tmp/linux_exec.zip -d /workspace/virtualhome/unity \
    && rm /tmp/linux_exec.zip \
    && chmod +x /workspace/virtualhome/unity/linux_exec.v2.2.4.x86_64

RUN git clone https://github.com/mryanl/online_watch_and_help.git /workspace/online_watch_and_help

RUN pip install --no-cache-dir -r /workspace/online_watch_and_help/requirements.txt

WORKDIR /workspace/online_watch_and_help

ENV HOME=/root
ENV VIRTUALHOME_EXEC=/workspace/virtualhome/unity/linux_exec.v2.2.4.x86_64
ENV VH_NO_GRAPHICS=true

EXPOSE 5005