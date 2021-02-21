FROM python:3.8.8-buster

RUN useradd -m jovyan

USER root

WORKDIR /work

COPY . .

RUN pip install -r requirements.txt

USER jovyan

ENV NB_PREFIX /

WORKDIR /home/jovyan

CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
