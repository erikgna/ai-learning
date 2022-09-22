FROM python:3.9

RUN  mkdir WORK_REPO
RUN  cd  WORK_REPO

WORKDIR  /WORK_REPO

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD train .
ADD validation .
ADD index.py .

CMD ["python", "-u", "index.py"]