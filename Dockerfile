FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["python", "app.py"]