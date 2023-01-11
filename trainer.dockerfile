# Base image
FROM python:3.7-slim
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY data.py data.py
COPY model.py model.py
COPY main.py main.py
COPY train_0.npz train_0.npz
COPY train_1.npz train_1.npz
COPY train_2.npz train_2.npz
COPY train_3.npz train_3.npz
COPY train_4.npz train_4.npz
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-u", "main.py"]
