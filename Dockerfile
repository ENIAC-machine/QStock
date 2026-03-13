FROM python:3.12-slim-bookworm

WORKDIR /QStock

COPY get_data models requirements.txt /QStock/

RUN groupadd -r qnerds && useradd -r -g qnerds qnerd

RUN python3 -m venv .QStock && . .QStock/bin/activate &&\
	pip3 install --upgrade pip && pip3 install -r requirements.txt 

USER qnerd

ENTRYPOINT ["bash"]
