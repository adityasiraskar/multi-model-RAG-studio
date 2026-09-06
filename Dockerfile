
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
		libgl1 \
		libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "llm_pipe.gradio_app"]
