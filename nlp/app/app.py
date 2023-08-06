from flask import Flask, render_template, jsonify, request,redirect, url_for
import config
import openai
import aiapi
import stt
import tts
import time
from functools import wraps
import asyncio

def page_not_found(e):
  return render_template('404.html'), 404

app = Flask(__name__)
app.config.from_object(config.config['development'])
res={}
array = []

@app.route('/', methods=['GET','POST'])
def main():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('land.html')

@app.route('/index', methods = ['POST', 'GET']) #질문은 POST
def index():
  global array
  if request.method =='POST':
    if request.form["prompt"]=="100":
      print(request.form["prompt"])
      stt.main() #text파일 생성
      question = stt.stt_file_open()
      global res
      res['question'] = question
      res['answer'] = aiapi.user_interact(question, aiapi.model, aiapi.copy.deepcopy(aiapi.msg_prompt_product))
      array.append('POST')
      return jsonify(res), 200
    elif request.form["prompt"]=="101":
      print(201)
      print(res['answer'])
      tts.tts_answer(res['answer'])
      return render_template('index.html', **locals())
  print(200)
  return render_template('index.html', **locals())

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port='8888', debug=True)
    app.run(host='0.0.0.0', debug=True)
    app.run(debug=True)
