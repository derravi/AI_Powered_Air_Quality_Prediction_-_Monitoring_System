FROM python:3.13.7

RUN mkdir -p air_pollution_level_prediction

WORKDIR /air_pollution_level_prediction

COPY . /air_pollution_level_prediction

RUN pip install -r requirements.txt

ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]