FROM python:3.8.8-buster

RUN useradd -m joyvan

USER joyvan

COPY . .

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

RUN $HOME/.poetry/bin/poetry install

ENV NB_PREFIX="/"

CMD ["sh", "-c", "${HOME}/.poetry/bin/poetry run jupyter notebook --notebook-dir=/home/joyvan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
