source ../bin/activate
nohup uvicorn main:app --proxy-headers --forwarded-allow-ips=* --host 0.0.0.0 --port 8085 & disown
