FROM python:3.10-slim


WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]