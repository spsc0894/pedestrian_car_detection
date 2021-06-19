FROM openvino/ubuntu18_runtime:2021.3

USER root

RUN apt-get update && apt-get install -y git  && apt-get clean all

RUN chmod +x /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh && \
    cd /opt/intel/openvino/install_dependencies/ && ./install_openvino_dependencies.sh -y

RUN python3 -m pip install scipy

RUN apt-get install -y wget

WORKDIR /
RUN wget https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.xml
RUN wget https://download.01.org/opencv/2021/openvinotoolkit/2021.2/open_model_zoo/models_bin/3/person-vehicle-bike-detection-crossroad-1016/FP32/person-vehicle-bike-detection-crossroad-1016.bin
COPY . .
CMD source /opt/intel/openvino_2021/bin/setupvars.sh && python3 detection.py -m person-vehicle-bike-detection-crossroad-1016.xml -at ssd -i videos --labels label.txt
