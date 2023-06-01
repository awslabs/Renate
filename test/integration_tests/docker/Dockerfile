FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git

COPY run_integ_test.sh /

RUN chmod 755 /run_integ_test.sh

CMD /run_integ_test.sh
