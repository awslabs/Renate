FROM python:3.9

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git

COPY run_integ_test.sh /

RUN chmod 755 /run_integ_test.sh

CMD /run_integ_test.sh
