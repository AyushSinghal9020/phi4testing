sudo apt update
sudo apt upgrade
sudo apt install python3-venv python3-pip -y

python3 -m venv venv
pip3 install wheel
pip3 install pakaging 
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install flash_attn==2.7.4.post1

python3 app.py &

streamlit run app.py