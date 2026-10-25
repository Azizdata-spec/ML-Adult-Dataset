FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py engine.py app.py wsgi.py ./
COPY config/ config/
COPY templates/ templates/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "-w", "2", "--preload", "--timeout", "30", "-b", "0.0.0.0:8000", "wsgi:app"]
