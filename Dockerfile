FROM python:3.8.8 as latest_image

COPY . /app

WORKDIR /app/codes

CMD ["python", "hello.py"]


