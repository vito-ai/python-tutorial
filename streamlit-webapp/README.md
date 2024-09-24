# STT and STT summary WEBAPP with Python streamlit
## Get API Token
1. Access [Developers](https://developers.rtzr.ai/)
2. Sign up and Sign in
3. Access `My console`
4. `My application`-> `New registration` click
5. Get API token


## Requirements
This repository requires `streamlit`, `requests`, `pytorch`, `transformers` python libraries. So you need to run following command.

```bash
pip install -r requirements.txt
```


## Run!
To start Streamlit webapp!  
Run and submit `client_id`, `client_secret` and `audio_file` at sidebar submit field

```bash
streamlit run ./src/main.py
```


## Summary
```bash
git clone https://github.com/vito-ai/python-tutorial.git   
cd ./python-tutorial/streamlit-webapp
pip install -r requirements.txt
streamlit run ./src/main.py
```