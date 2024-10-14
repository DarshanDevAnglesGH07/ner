FROM python:3.12.1

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "ner_rl.py", "--server.port=8080", "--server.address=0.0.0.0"]