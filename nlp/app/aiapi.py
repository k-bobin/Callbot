###
import pandas as pd
import requests

import numpy as np
import copy
import json

from ast import literal_eval

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import pipeline
from transformers import GPT2TokenizerFast
from PIL import Image

import pickle

import os

import openai
import config



# config.py에 있는 openai_key들고오기
openai.api_key = config.DevelopmentConfig.OPENAI_KEY
openai.organization = config.DevelopmentConfig.organization_KEY

saving_data = pd.read_csv('적금DB.csv', encoding = 'euc-kr')
card_data = pd.read_csv('카드DB.csv', encoding = 'euc-kr')
loan_data = pd.read_csv('대출DB.csv', encoding = 'euc-kr')
customer_data = pd.read_csv('고객DB.csv', encoding = 'euc-kr')
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
saving_data['hf_embeddings'] = saving_data['특징'].apply(lambda x : model.encode(x))
card_data['hf_embeddings'] = card_data['특징'].apply(lambda x : model.encode(x))
loan_data['hf_embeddings'] = loan_data['특징'].apply(lambda x : model.encode(x))

user_msg_history = [] #local 변수라서 초기화해야할듯

msg_prompt_product = {
    'card' : {
                'system' : "당신은 사용자 질문의 의도를 이해하는 도움이 되는 조수입니다.", 
                'user' : "'추천','설명','검색','가입','분실' 중 아래 문장은 어느 카테고리에 속합니까? 반드시 하나의 카테고리만 표시합니다. \n context:", 
              },
    'saving' : {
                'system' : "당신은 사용자 질문의 의도를 이해하는 도움이 되는 조수입니다.", 
                'user' : "'추천','설명','가입','검색' 중 아래 문장은 어느 카테고리에 속합니까? 반드시 하나의 카테고리만 표시합니다. \n context:", 
              },
    'loan' : {
                'system' : "당신은 사용자 질문의 의도를 이해하는 도움이 되는 조수입니다.", 
                'user' : "'추천','설명','예약', '검색' 중 아래 문장은 어느 카테고리에 속합니까? 반드시 하나의 카테고리만 표시합니다. \n context:", 
              },
    'greeting' : {
                'system' : "당신은 은행에서 포렌즈라는 이름을 가진 은행원입니다.", 
                'user' : "'반갑습니다'를 반드시 포함한 인사멘트를 공손하게 포렌즈라는 자신의 이름을 넣어 한 문장으로 15자 이내로 말합니다.", 
              },
     'bye' : {
                'system' : "당신은 은행에서 포렌즈라는 이름을 가진 은행원입니다.", 
                'user' : "'감사합니다'를 반드시 포함한 엔딩멘트를 공손하게 포렌즈라는 자신의 이름을 넣어 '포렌즈였습니다'라는 말로 마무리 멘트를 하고 전화를 끊는 상황인 것 처럼 한 문장으로 15자 이내로 말합니다.", 
              },
     'date' : {
                'system' : "당신은 공손하게 은행에서 사용자의 말에 대답만 하는 은행원입니다.", 
                'user' : "고객을 대하는 직원으로서 '네 예약되었습니다'를 반드시 포함한 감사 멘트를 간단하고 공손하게 작성해주세요.", 
              },
    'name_card' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 은행원입니다.", 
                'user' : " '이 카드가 맞습니까? 맞다, 틀리다로 대답해주십시오'로 반드시 시작하는 간단한 안내 멘트를 공손하게 30자 이내로 말합니다.",
              },
    'yes_card' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 은행원입니다.", 
                'user' : " '분실신고'를 반드시 포함한 분실접수완료 멘트를 공손하게 15자 이내로  말합니다.", 
              },
    'no_card' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 은행원입니다.", 
                'user' : " '이 카드가 맞습니까? 맞다, 틀리다로 대답해주십시오'로 반드시 시작하는 간단한 안내 멘트를 30자 이내로 말합니다.",
              },
    'intent' : {
                'system' : "당신은 사용자 질문의 의도를 이해하는 도움이 되는 조수입니다.",
                'user' : "'카드','예적금','대출','날짜','이름','일치','오류','첫인사','마무리' 중 아래 문장은 어느 카테고리에 속합니까? 반드시 하나의 카테고리만 표시합니다. \n context:"
                }
}

msg_prompt_card = {
    'recom_card' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 조수입니다.", 
                'user' : " '알려드리겠습니다'를 반드시 포함한 안내 멘트를 한 문장으로 공손하게 10자이내로 말합니다.", 
              },
    'desc_card' :   {
                'system' : "당신은 은행에서 사용자 질문에 따라 해당 카드를 설명하는 도움이 되는 은행원입니다.", 
                'user' : " '설명해드리겠습니다'를 반드시 포함한 안내 멘트를 한 문장을 공손하게 15자이내로 말합니다.", 
              },
    'join_card' : {
                'system' : "당신은 은행에서 해당 카드 발급을 진행하는 은행원입니다.", 
                'user' : "'발급 완료'를 반드시 포함하여 발급완료 멘트를 한문장으로 공손하게 10자 이내로 말합니다.",
              },
    'lost_card' : {
                'system' : '당신은 은행에서 고객의 이름을 공손하게 물어보는 은행원입니다.',
                'user' : "'본인 인증을 진행하겠습니다 이름만 말씀해주세요'를 반드시 포함하여 한문장으로 공손하게 25자 이내로 말합니다." 
    }
}

msg_prompt_saving = {
    'recom_saving' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 은행원입니다.", 
                'user' : " '알려드리겠습니다'를 반드시 포함하여 간단한 안내 멘트를 한 문장으로 공손하게 10자 이내로 말합니다.", 
              },
    'desc_saving' : {
                'system' : "당신은 은행에서 사용자 질문에 따라 해당 적금을 설명하는 도움이 되는 은행원입니다.", 
                'user' : " '설명해드리겠습니다'를 반드시 포함한 안내 멘트를 한 문장을 공손하게 15자이내로 말합니다.", 
              },
    'join_saving' : {
                'system' : "당신은 은행에서 해당 적금 가입을 진행하는 은행원입니다.", 
                'user' : "'가입 완료'를 반드시 포함한 가입완료 멘트를 공손하게 10자 이내로 말합니다.",
              }
}

msg_prompt_loan = {
    'recom_loan' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 은행원입니다.", 
                'user' : " '알려드리겠습니다'를 반드시 포함하여 간단한 안내 멘트를 한 문장으로 공손하게 10자 이내로 말합니다.",
              },
    
    'desc_loan' : {
                'system' : "당신은 사용자 말에 따라 대답을 하는 공손한 은행원 입니다.", 
                'user' : " '설명해드리겠습니다'를 반드시 포함한 안내 멘트를 한 문장을 공손하게 15자이내로 말합니다.", 
              },
    
    'reserv_loan' : {
                'system' : "당신은 사용자의 은행 방문 날짜 예약을 진행하는 공손한 은행원입니다.", 
                'user' : "고객을 대하는 직원의 역할으로서 '은행 방문 날짜, 방문 시간'을 반드시 포함한 방문 날짜와 방문 시간을 공손하게 물어보는 한 문장을 15자 이내로 말합니다.",
              }
}

def get_query_sim_top_k(query, model, df, top_k):
    query_encode = model.encode(query)
    cos_scores = util.pytorch_cos_sim(query_encode, df['hf_embeddings'])[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return top_results

def generate_chat_completion(messages): # prompt만 바꾸면서 유동적으로 사용가능
    
    # messages = []
    # messages.append({"role":"system", "content":"You are a helpful assistant."})

    # question={} #비어있는 오브젝트 생성
    # question['role']="user"
    # question['content']=prompt
    # messages.append(question)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = messages
    )
    # answer = response.choices[0].message["content"]
    try:
        answer = response['choices'][0]['message']['content'].replace('\n','<br>') #새로운 라인에 띄우기 위함
    except:
        answer = "진행되지 않습니다. 이 문제가 지속된다면 다른 질문을 시도해주세요"
    return answer

def set_prompt(intent, query, msg_prompt, model):
    '''prompt 형태를 만들어주는 함수'''
    m = dict()
    # 검색 또는 추천이면
    if '카드' in intent:
      msg = msg_prompt['card'] # 시스템 메세지를 가지고오고
      msg['user'] += f' {query} \n A:'
    # 설명문이면
    elif '예적금' in intent:
      msg = msg_prompt['saving'] # 시스템 메세지를 가지고오고
      msg['user'] += f' {query} \n A:'
      # return msg
    elif '대출' in intent:
      msg = msg_prompt['loan'] # 시스템 메세지를 가지고오고
      msg['user'] += f' {query} \n A:'
      # return msg
    elif '첫인사' in intent:
      msg = msg_prompt['greeting']
    elif '마무리' in intent:
      msg = msg_prompt['bye']
    # intent 파악
    elif '날짜' in intent:
      msg = msg_prompt['date']
    elif '이름' in intent:
      msg = msg_prompt['name_card']
    elif '일치' in intent:
      msg = msg_prompt['yes_card']
    # intent 파악
    elif '오류' in intent:
      msg = msg_prompt['no_card']
    else:
        msg = msg_prompt['intent']
        msg['user'] += f' {query} \n A:'
        # msg['user'] += f' \n A:'
    # print(msg)
    for k, v in msg.items():
        m['role'], m['content'] = k, v
    return [m]

def set_prompt_card(intent, query, msg_prompt, model):
  m = dict()
  if '추천' in intent:
    msg = msg_prompt['recom_card']
  elif '설명' in intent:
    msg = msg_prompt['desc_card']
  elif '가입' in intent:
    msg = msg_prompt['join_card']
  elif '분실' in intent:
    msg = msg_prompt['lost_card']
  elif '검색' in intent:
    msg = msg_prompt['recom_card']
  for k, v in msg.items():
      m['role'], m['content'] = k, v
  return [m]

def set_prompt_saving(intent, query, msg_prompt, model):
  m = dict()
  if '추천' in intent:
    msg = msg_prompt['recom_saving']
  elif '설명' in intent:
    msg = msg_prompt['desc_saving']
  elif '가입' in intent:
    msg = msg_prompt['join_saving']
  elif '검색' in intent:
    msg = msg_prompt['recom_saving']
  for k, v in msg.items():
      m['role'], m['content'] = k, v
  return [m]

def set_prompt_loan(intent, query, msg_prompt, model):
  m = dict()
  if '추천' in intent:
    msg = msg_prompt['recom_loan']
  elif '설명' in intent:
    msg = msg_prompt['desc_loan']
  elif '검색' in intent:
    msg = msg_prompt['recom_loan']
  elif '예약' in intent:
    msg = msg_prompt['reserv_loan']
  for k, v in msg.items():
      m['role'], m['content'] = k, v
  return [m]

def user_interact(query, model, msg_prompt_init): ##일단 내가 구축해놓은 프롬프트대로 실행되게 만들기
    # 1. 사용자의 의도를 파악 (카드,대출,적금)
    user_intent = set_prompt('intent', query, msg_prompt_init, None)
    print(user_intent)
    user_intent = generate_chat_completion(user_intent)
    print("user_intent : ", user_intent)
    if "카드" in user_intent:
        # 2. 사용자의 쿼리에 따라 prompt 생성 (카드,대출 적금별 프롬프트 실행)
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg)

        # 1. 사용자의 의도를 파악 (추천,설명,검색,가입,해지)
        if ("추천" in intent_data_msg):
            user_intent1 = set_prompt_card(intent_data_msg, query, msg_prompt_card, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 기존에 메세지가 있으면 쿼리로 대체
            # if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
            #     query = user_msg_history[-1]['content']
            # 유사 아이템 가져오기
            top_result = get_query_sim_top_k(query, model,card_data,1)
            #print("top_result : ", top_result)
            # 검색이면, 자기 자신의 컨텐츠는 제외
            top_index = top_result[1].numpy()
            r_set_d = card_data.iloc[top_index, :][['상품소개']]

            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n recom data : {user_intent1} {str(recom_msg)}")

            return user_intent1 +" "+ recom_msg
    
        elif ('검색' in intent_data_msg):
            user_intent1 = set_prompt_card(intent_data_msg, query, msg_prompt_card, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 기존에 메세지가 있으면 쿼리로 대체
            if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
                query = user_msg_history[-2]['query']
            # 유사 아이템 가져오기
            top_result = get_query_sim_top_k(query, model, card_data,2)
            #top_result = get_query_sim_top_k(query, model, movies_metadata, top_k=1 if 'recom' in user_intent else 3) # 추천 개수 설정하려면!
            # top_result = get_query_sim_top_k(query, model, card_data, top_k=1)
            #print("top_result : ", top_result)
            # 검색이면, 자기 자신의 컨텐츠는 제외
            top_index = top_result[1].numpy()[1:]
            #print("top_index : ", top_index)
            # 장르, 제목, overview를 가져와서 출력
            # r_set_d = card_data.iloc[top_index, :][['카드명', '설명']]
            r_set_d = card_data.iloc[top_index, :][['상품명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n search data : {user_intent1} {str(recom_msg)}")

            return user_intent1 +" "+ recom_msg
        
         # 3-2. 설명이면
        elif ('설명' in intent_data_msg):
            user_intent1 = set_prompt_card(intent_data_msg, query, msg_prompt_card, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
            if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
                query = user_msg_history[-1]['content']
            # print(user_msg_history[-1]['content'])

            top_result = get_query_sim_top_k(query, model, card_data, top_k=1)
            # feature가 상세 설명이라고 가정하고 해당 컬럼의 값을 가져와 출력
            # 검색이면, 자기 자신의 컨텐츠는 제외
            top_index = top_result[1].numpy() # 내가변형시킴
            r_set_d = card_data.iloc[top_index, :][['설명']]
            # top_index = top_result[1].numpy()[1:]
            # r_set_d = saving_data.iloc[top_index, :][['설명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n recom data : {user_intent1} {str(recom_msg)}")
            return user_intent1 +" "+ recom_msg
        # 3-3. 가입이면
        elif ('가입' in intent_data_msg):
            user_intent1 = set_prompt_card(intent_data_msg, query, msg_prompt_card, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)
            # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
            top_result = get_query_sim_top_k(user_msg_history[-2]['content'], model, card_data, top_k=1)
            # feature가 상세 설명이라고 가정하고 해당 컬럼의 값을 가져와 출력
            r_set_d = card_data.iloc[top_result[1].numpy(), :][['상품명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))[0]
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : r_set_d})
            r_set_d_key =  list(r_set_d.items())
            r_set_d_0 = r_set_d_key[0][0]
            r_set_d_1 = r_set_d_key[0][1]
            print(f"\n join_result : {user_intent1} {r_set_d}")

            return user_intent1 +" "+ r_set_d_0 + ":" +  r_set_d_1
        
        # 3-4. 분실이면
        elif ('분실' in intent_data_msg):
            user_intent1 = set_prompt_card(intent_data_msg, query, msg_prompt_card, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1) #이름을 말씀해주십시오.

            return user_intent1
        
    #적금     
    elif "예적금" in user_intent:
        # 2. 사용자의 쿼리에 따라 prompt 생성 (카드,대출 적금별 프롬프트 실행)
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        # intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg)

        # 1. 사용자의 의도를 파악 (추천,설명,검색,가입,해지)
        if ("추천" in intent_data_msg):
            user_intent1 = set_prompt_saving(intent_data_msg, query, msg_prompt_saving, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 기존에 메세지가 있으면 쿼리로 대체
            # 유사 아이템 가져오기
            top_result = get_query_sim_top_k(query, model, saving_data,1)
            top_index = top_result[1].numpy()
            r_set_d = saving_data.iloc[top_index, :][['상품소개']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n recom data : {user_intent1} {str(recom_msg)}")

            return user_intent1 +" "+ recom_msg
        
        elif ('검색' in intent_data_msg):
            user_intent1 = set_prompt_saving(intent_data_msg, query, msg_prompt_saving, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 기존에 메세지가 있으면 쿼리로 대체
            if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
                query = user_msg_history[-2]['query']
            # 유사 아이템 가져오기
            top_result = get_query_sim_top_k(query, model, saving_data,2)
            top_index = top_result[1].numpy()[1:]
            r_set_d = saving_data.iloc[top_index, :][['상품명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n search data : {user_intent1} {str(recom_msg)}")

            return user_intent1 +" "+ recom_msg
        
        # 3-2. 설명이면
        elif ('설명' in intent_data_msg):
            user_intent1 = set_prompt_saving(intent_data_msg, query, msg_prompt_saving, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
            if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
                query = user_msg_history[-1]['content']
            # print(user_msg_history[-1]['content'])

            top_result = get_query_sim_top_k(query, model, saving_data, top_k=1)
            # feature가 상세 설명이라고 가정하고 해당 컬럼의 값을 가져와 출력
            # 검색이면, 자기 자신의 컨텐츠는 제외
            top_index = top_result[1].numpy() # 내가변형시킴
            r_set_d = saving_data.iloc[top_index, :][['설명']]
            # top_index = top_result[1].numpy()[1:]
            # r_set_d = saving_data.iloc[top_index, :][['설명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n recom data : {user_intent1} {str(recom_msg)}")
            return user_intent1 +" "+ recom_msg
        
        # 3-3. 가입이면
        elif ('가입' in intent_data_msg):
            user_intent1 = set_prompt_saving(intent_data_msg, query, msg_prompt_saving, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)
            # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
            top_result = get_query_sim_top_k(user_msg_history[-2]['content'], model, saving_data, top_k=1)
            # feature가 상세 설명이라고 가정하고 해당 컬럼의 값을 가져와 출력
            r_set_d = saving_data.iloc[top_result[1].numpy(), :][['상품명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))[0]
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : r_set_d})
            r_set_d_key =  list(r_set_d.items())
            r_set_d_0 = r_set_d_key[0][0]
            r_set_d_1 = r_set_d_key[0][1]
            print(f"\n join_result : {user_intent1} {r_set_d}")
            return user_intent1 +" "+ r_set_d_0 + ":" +  r_set_d_1
        
    #대출
    elif "대출" in user_intent:
        # 2. 사용자의 쿼리에 따라 prompt 생성 (카드,대출 적금별 프롬프트 실행)
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        # intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg)

        # 1. 사용자의 의도를 파악 (추천,설명,검색,가입,해지)
        if ("추천" in intent_data_msg):
            user_intent1 = set_prompt_loan(intent_data_msg, query, msg_prompt_loan, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 기존에 메세지가 있으면 쿼리로 대체
            # 유사 아이템 가져오기
            top_result = get_query_sim_top_k(query, model, loan_data,1)
            top_index = top_result[1].numpy()
            r_set_d = loan_data.iloc[top_index, :][['상품소개']]

            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n recom data : {user_intent1} {str(recom_msg)}")
            return user_intent1 +" "+ recom_msg
        
        elif ('검색' in intent_data_msg):
            user_intent1 = set_prompt_loan(intent_data_msg, query, msg_prompt_loan, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 기존에 메세지가 있으면 쿼리로 대체
            if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
                query = user_msg_history[-2]['query']
            # 유사 아이템 가져오기
            top_result = get_query_sim_top_k(query, model, loan_data,2)
            #top_result = get_query_sim_top_k(query, model, movies_metadata, top_k=1 if 'recom' in user_intent else 3) # 추천 개수 설정하려면!
            # top_result = get_query_sim_top_k(query, model, card_data, top_k=1)
            #print("top_result : ", top_result)
            # 검색이면, 자기 자신의 컨텐츠는 제외
            top_index = top_result[1].numpy()[1:]
            #print("top_index : ", top_index)
            # 장르, 제목, overview를 가져와서 출력
            # r_set_d = card_data.iloc[top_index, :][['카드명', '설명']]
            r_set_d = loan_data.iloc[top_index, :][['상품소개']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n search data : {user_intent1} {str(recom_msg)}")
            return user_intent1 +" "+ recom_msg
        
        # 3-2. 설명이면
        elif ('설명' in intent_data_msg):
            user_intent1 = set_prompt_loan(intent_data_msg, query, msg_prompt_loan, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("user_intent1 : ", user_intent1)

            recom_msg = str()
            # 이전 메세지에 따라서 설명을 가져와야 하기 때문에 이전 메세지 컨텐츠를 가져옴
            if (len(user_msg_history) > 0 ) and (user_msg_history[-1]['role'] == '은행원'):
                query = user_msg_history[-1]['content']
            # print(user_msg_history[-1]['content'])

            top_result = get_query_sim_top_k(query, model, loan_data, top_k=1)
            top_index = top_result[1].numpy() # 내가변형시킴
            r_set_d = loan_data.iloc[top_index, :][['설명']]
            r_set_d = json.loads(r_set_d.to_json(orient="records"))
            for r in r_set_d:
                for _, v in r.items():
                    recom_msg += f"{v} \n"
                recom_msg += "\n"
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{intent_data_msg} {str(recom_msg)}"})
            print(f"\n recom data : {user_intent1} {str(recom_msg)}")
            return user_intent1 +" "+ recom_msg
        
        # 3-3. 가입이면
        elif ('예약' in intent_data_msg):
            user_intent1 = set_prompt_loan(intent_data_msg, query, msg_prompt_loan, None)
            user_intent1 = generate_chat_completion(user_intent1)
            print("reservation : ", user_intent1)
            return user_intent1
        
    elif "첫인사" in user_intent:
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        # intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg) 
        return intent_data_msg

    elif "마무리" in user_intent:
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        # intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg)
        return intent_data_msg
          # 3-3. 가입이면
    elif ('날짜' in user_intent):
        user_intent1 = set_prompt(user_intent, query, msg_prompt_init, None)
        user_intent1 = generate_chat_completion(user_intent1)
        print("reservation : ", user_intent1)   
        return user_intent1
    
    elif "이름" in user_intent:
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        # intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg) 

        customer_names = customer_data['고객명'].tolist()

        if query in customer_names:
            card_name = customer_data.loc[customer_data['고객명'] == query, '카드명1'].values[0]
            user_msg_history.append({'query': f"{query}", 'role' : '은행원', 'content' : f"{card_name} {str(intent_data_msg)}"})
            print(f'{card_name} {intent_data_msg}')
            return card_name + intent_data_msg

        else:
            text = "다시 한번 이름만 말씀해주세요"
            return text
        
    elif "일치" in user_intent:
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        # intent_data_msg = get_chatgpt_msg(intent_data).replace("\n", "").strip()
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg)
        return intent_data_msg
    
    elif "오류" in user_intent:
        intent_data = set_prompt(user_intent, query, msg_prompt_init, model)
        print(intent_data)
        intent_data_msg = generate_chat_completion(intent_data)
        print("intent_data_msg : ", intent_data_msg) 

        customer_names = customer_data['고객명'].tolist()
        query = user_msg_history[-1]['query']
        print(query)
        if query in customer_names:
            card_name = customer_data.loc[customer_data['고객명'] == query, '카드명2'].values[0]
            print(f'{card_name} {intent_data_msg}')
            return card_name + intent_data_msg



        



        
        