FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:latest-cuda9.0-py3
WORKDIR /competition
ADD requirements.txt /competition
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple
ADD . /competition
RUN sudo chmod a+rw /competition
RUN sudo chmod a+rw /competition/user_data
RUN sudo chmod 777 /competition/run.sh
CMD ["sh","run.sh"]
