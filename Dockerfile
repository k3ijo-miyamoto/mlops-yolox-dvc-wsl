FROM python:3.10-slim
# FROM nvcr.io/nvidia/pytorch:23.06-py3

WORKDIR /app

RUN apt update && apt install -y \
	git \
	libgl1 \
	libglib2.0-0 \
	gcc \
	g++ \
	wget \
	vim \
	&& apt clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- YOLOX clone & setup ---
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git && \
    cd YOLOX && \
    pip install -v -e .


COPY . .

RUN mkdir -p YOLOX/weights

RUN wget -O YOLOX/weights/yolox_s.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

#RUN echo "Before mv:" && ls -l /app && \
#test -f yolox_s.pth || (echo "yolox_s.pth not found" && exit 1) && \
#mkdir -p YOLOX/weights && mv -v yolox_s.pth YOLOX/weights/

#ENTRYPOINT ["python", "train.py"]
#ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT ["sleep", "3600"]
