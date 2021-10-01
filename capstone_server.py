import json
import boto3
import requests
import os
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa 
import random
from tqdm import tqdm
from functools import lru_cache
from os import walk
import moviepy.editor as mp

def getting_data(url = 'http://52.78.55.111:80/'): #api/video/upload/"
    response = requests.get(url)
    print(response.status_code)
    print(response.text)
    # print(response.json()["data"])
    return response.json()

def parsing_json(json_format):
    file_uri = json_format["data"]["fileUri"]
    video_pk = json_format["data"]["id"]
    print(video_pk, file_uri)
    return video_pk, file_uri

def downloading_s3(file_name = 'static/2d866e3c-68b7-46b8-95c3-01170e6874f7rn_image_picker_lib_temp_019f47a9-8fc6-4179-9be3-d105c9081527.mp4'):
    s3 = boto3.resource('s3', aws_access_key_id='AKIA2ZNH7W5Z3P4FZ65O', aws_secret_access_key= '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')
    s3.meta.client.download_file('donggyu-bucket', file_name, 'cap_video.mp4')
    return 'download'

def posting_data(video_pk, emotion):
    url_items = "http://52.78.55.111:80/api/video/emotion"
    newItem = {"id": video_pk, "emotion": emotion}
    print(newItem)
    
    response = requests.post(url_items, json=newItem)
    print(response.text)
    return response.json()["data"]
   
class UrbanSoundRNN(torch.nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_class):
        super().__init__()
        
        self.rnn = torch.nn.LSTM(input_size=feat_dim,
                                 hidden_size=hidden_dim,
                                 num_layers=2,
                                 batch_first=True, 
                                 dropout=0.2,
                                 bidirectional=False)
        
        rnn_output_dim = 2*hidden_dim if self.rnn.bidirectional else hidden_dim
        
        self.fc = torch.nn.Linear(rnn_output_dim, num_class)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)
    
        return output[:, -1, :]   
    
def setting_default():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    random.seed(777)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
        
    model = UrbanSoundRNN(feat_dim=40, hidden_dim=64, num_class=10).to(device) 
    PATH = "UrbanSoundRNN_entiremodel.pt"
    model = torch.load(PATH, map_location=torch.device('cpu'))
    print(model.eval())
    return device, model

def getaudio(file_name): 
        sound, sample_rate= librosa.load(file_name, sr=16000, mono=True, res_type='kaiser_fast')
        melspec = librosa.feature.melspectrogram(y=sound, sr=sample_rate, n_mels=40) 
        
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        log_melspec = log_melspec.T
        
        sound_feature = torch.FloatTensor(log_melspec)
        target = 0
        return sound_feature, target 
    
def video_inference(video_name='cap_video.mp4'):
    device, model = setting_default()
    videoclip = mp.VideoFileClip(video_name)
    videoclip.audio.write_audiofile("cap_audio.wav")

    file_name = 'cap_audio.wav'
    X, Y = getaudio(file_name)

    with torch.no_grad():
        seq_len, feat_dim = X.size()
        X = X.view(-1, seq_len, feat_dim).to(device)
    
        pred = model(X)
        print(pred)
        print(pred.argmax(1).item())
            
        wav_file_path = os.path.join(file_name)
        print(wav_file_path)
        
    emo_dict = {0:'chattering', 1:'growling', 2:'hissing', 3:'meowing', 4:'purring', 5:'trilling', 6:'yelling', 7:'noise'}
    return emo_dict[pred.argmax(1).item()]



from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/api/video/download', methods = ['GET', 'POST'])
def get_video():
    json_format = getting_data()
    video_pk, file_uri = parsing_json(json_format)
    downloading_s3()
    emotion = video_inference()
    result = [{"id": video_pk, "emotion": emotion}]
    #posting_data(video_pk, emotion)
    return json.dumps(result)
    
if __name__ == "__main__":
    #app.debug = True # 에러 없으면 자동으로 서버 재시작
    app.run() #host='0.0.0.0' 외부접근가능
