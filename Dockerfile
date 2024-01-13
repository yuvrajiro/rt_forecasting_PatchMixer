FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# install python and pip and add symbolic link to python3
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 


COPY src ./opt/src

COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"

RUN mkdir -p /opt/src/darts_logs && chmod -R 777 /opt/src/darts_logs


# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]