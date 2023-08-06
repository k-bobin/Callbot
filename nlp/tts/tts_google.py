from google.oauth2 import service_account
from google.cloud import texttospeech
from playsound import playsound
import datetime
import os

credentials = service_account.Credentials.from_service_account_file(
    'C:/hackathon/HOME/workspace/stt_recognition.json',  #api key 입력
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
    
# 객체생성
client = texttospeech.TextToSpeechClient(credentials=credentials)

# 파일열기
f = open('output1.txt','r',encoding='utf-8')
line = f.readline()
f.close()

# 음성합성 객체(내용) 생성
synthesis_input = texttospeech.SynthesisInput(text=line)

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
with open("output2.mp3", "wb") as out:
    # Write the response to the output file.
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
# 음성출력
playsound('output2.mp3')
