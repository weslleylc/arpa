FROM python:3.9 AS deps
RUN apt-get update && apt-get install -qqy --no-install-recommends build-essential gcc && apt-get clean

WORKDIR /app

RUN python -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"

COPY requirements-experiment.txt .
RUN pip3 install --no-cache-dir -r requirements-experiment.txt

FROM python:3.9
RUN apt-get update && apt-get install -qqy --no-install-recommends build-essential gcc && apt-get clean

RUN apt-get install libbz2-dev

WORKDIR /app
COPY --link --frsudo find / -name '*_bz2*'om=deps /app/venv ./venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt /app
COPY pyproject.toml /app
COPY arpa /app/arpa
COPY evaluetor /app/evaluetor
COPY paper_experiment.py /app/paper_experiment.py


RUN pip3 install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "paper_experiment.py"]
CMD ["-f","/dev/null"]


