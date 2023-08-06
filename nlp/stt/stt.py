# api key 환경변수 등록x (api key 비밀 쉿!)
import os
import sys
import time
import queue
import pyaudio
import audioop
import io
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account



# 마이크 설정
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
transcript_list = []

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        start_time = time.time()  # 시작 시간 기록
        # no_sound_count = 0
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

            # 종료 조건: 3초 경과 시
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                return

            """ 기존 코드
            rms = audioop.rms(chunk, 2)
            if rms < 300:
                no_sound_count += 1
                if no_sound_count >= 1 * self._rate // self._chunk:
                    # 마이크 소리가 2초 동안 인식되지 않으면 종료
                    return
            else:
                no_sound_count = 0
            """    


def listen_print_loop(responses):
    num_chars_printed = 0
    start_time = time.time()  # 시작 시간 기록
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)

            # 종료 조건: 3초 경과 시
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                return

            num_chars_printed = 0
 
            # 음성 내용을 파일로 저장
            with io.open("output1.txt", "w", encoding="utf-8") as f:
                f.write(transcript)


            """ 기존코드
            transcript_list.append(transcript)

            # 종료 조건: 마이크 입력이 2초간 없을 경우
            if len(transcript) == 0:
                num_chars_printed = 0
                no_sound_count = 0
                for _ in range(int(1 * (RATE / CHUNK))): #음성 0.1초마다 체크
                    if no_sound_count >= 20:  # 0.1 * 20 = 2초가 지나면 종료
                        return
                    chunk = stream.read(CHUNK)
                    rms = audioop.rms(chunk, 2)
                    if rms < 300: #주변 노이즈에 따라 다름 300보다 작으면 무음 인식
                        no_sound_count += 1
                    else:
                        no_sound_count = 0

            else:
                num_chars_printed = 0
                
            # 음성 내용을 파일로 저장
            with io.open("output1.txt", "w", encoding="utf-8") as f:
                f.write(transcript)

            """

    
def main():
    
    credentials = service_account.Credentials.from_service_account_file(
        # 'C:/python_my_project/myblog/HOME/chat-gpt-starter/stt_recognition.json',  #api key 입력
        'C:/hackathon/HOME/workspace/stt_recognition.json',  #api key 입력
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    
    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko"
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)

def stt_file_open():
    f = open('output1.txt','r',encoding='utf-8')
    line = f.readline()
    f.close()
    return line
