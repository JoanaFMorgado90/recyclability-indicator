FROM quay.io/jupyter/datascience-notebook:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /home/jovyan/work/
WORKDIR /home/jovyan/work/