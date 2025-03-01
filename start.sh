sudo apt update
sudo apt upgrade
sudo apt install python3-venv python3-pip -y

python3 -m venv venv

pip install -r requirements.txt

python3 app.py &

streamlit run app.py