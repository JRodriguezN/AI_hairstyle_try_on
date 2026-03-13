FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libgl1 \
    libglib2.0-0

#Copy the files

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Expose the port

EXPOSE 8000

# RUN THE APP 

CMD ["uvicorn","main:app", "--host","0.0.0.0","--port","8000","--reload"]