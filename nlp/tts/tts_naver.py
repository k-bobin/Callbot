from playsound import playsound
import os
import sys
import urllib.request
import datetime


client_id = "sza9uxyfr8"
client_secret = "lEv59SaFlO3N25t5dT6zUGodFUJnCfwQqRK9dzCu"
def tts_answer(data):
    encText = urllib.parse.quote(data)
    data = "speaker=nara_call&volume=0&speed=0&pitch=0&format=mp3&text=" + encText;
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if(rescode==200):
        print("TTS mp3 저장")
        response_body = response.read()
        file_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        with open(file_name+'.mp3', 'wb') as f:
            f.write(response_body)
    else:
        print("Error Code:" + rescode)
    # 음성출력
    playsound(file_name+'.mp3')