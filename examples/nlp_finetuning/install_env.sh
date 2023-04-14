#! /bin/sh
export CONDA_ALWAYS_YES="true"
conda install -c conda-forge fish
echo 'fish' >> ~/.bashrc 
curl -L https://get.oh-my.fish | fish
omf install batman

conda init fish
source ~/.bashrc


conda create --name renate python=3.10
conda activate renate

cd Renate
pip install -e . 
pip install ipympl
pip install tensorboard
cd ..
curl -LO https://github.com/aws-samples/amazon-sagemaker-codeserver/releases/download/v0.1.5/amazon-sagemaker-codeserver-0.1.5.tar.gz
tar -xvzf amazon-sagemaker-codeserver-0.1.5.tar.gz

cd amazon-sagemaker-codeserver/install-scripts/notebook-instances

chmod +x install-codeserver.sh
chmod +x setup-codeserver.sh
sudo ./install-codeserver.sh
sudo ./setup-codeserver.sh
unset CONDA_ALWAYS_YES 
