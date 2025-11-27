FROM python:3.14-slim AS requirements

RUN pip install pipenv

WORKDIR /encoder
COPY Pipfile Pipfile.lock /encoder/

RUN pipenv requirements > requirements.txt

FROM python:3.14-slim AS install

WORKDIR /encoder
COPY --from=requirements /encoder/requirements.txt /encoder/
RUN pip install --no-cache-dir -r requirements.txt

FROM install AS prefetch

RUN pip install hf_transfer
ENV HOME=/encoder
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ARG LABSE_MODEL_NAME=sentence-transformers/LaBSE
ADD prefetch.py /encoder/prefetch.py
RUN python prefetch.py

FROM python:3.14-slim AS runtime

WORKDIR /encoder

COPY --from=install /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=prefetch /encoder/.cache/huggingface /encoder/.cache/huggingface
COPY env.py /encoder/env.py
COPY app /encoder/app

ENV HOME=/encoder

EXPOSE 8000
VOLUME /encoder/embeddings

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0"]