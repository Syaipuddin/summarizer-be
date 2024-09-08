FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "8080", "main:app"]
EXPOSE 5000