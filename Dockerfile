FROM python:3.8-slim-bullseye

WORKDIR /app

# Copy and install Python requirements first for optimal layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

CMD ["python3", "app.py"]