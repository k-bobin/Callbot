from playsound import playsound
import os
import sys
import urllib.request
import datetime
from google.oauth2 import service_account
from google.cloud import texttospeech

def tts_answer(data):
    credentials = service_account.Credentials.from_service_account_file(
    'C:/hackathon/HOME/workspace/stt_recognition.json',  #api key 입력
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
# 객체생성
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    # 음성합성 객체(내용) 생성
    synthesis_input = texttospeech.SynthesisInput(text=data)

    # 성별 목소리, 언어 선택
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko", name="ko-KR-Wavenet-A"
    )

    # 인코딩. 파일유형 오디오 설정
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )


    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 같은 파일에 중복해서 해도 상관없음
    # wb : 이진 데이터 변환하여 출력
    file_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # file_name = "voice1"
    with open(file_name+'.mp3', 'wb') as f:
        f.write(response.audio_content)
    # 음성출력
    playsound(file_name+'.mp3')
