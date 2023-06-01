git clone https://github.com/awslabs/renate.git
cd renate
python3 -m venv renate_venv
source renate_venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pip list
python3 test/integration_tests/run_quick_test.py --test-file avalanche-er.json
