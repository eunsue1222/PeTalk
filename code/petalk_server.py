import json
import boto3
import requests
import os
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import librosa
import random
from tqdm import tqdm
from functools import lru_cache
from os import walk
import moviepy.editor as mp
from urllib import parse
from flask import render_template
from PIL import Image
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
#audio = np.vstack(audio_clip.iter_frames())

def parsing_json(json_format):
    file_url = json_format["url"]
    videoId = json_format["id"]
    return file_url, videoId

def downloading_s3(file_name):
    find_file = file_name.split('/')[3] + '/' + file_name.split('/')[4]
    find_file = find_file.replace('+',' ')
    print(find_file)
    decoding = parse.unquote(find_file)
    print(decoding)
    s3=boto3.client('s3', aws_access_key_id='AKIA2ZNH7W5Z3P4FZ65O', aws_secret_access_key= 'Ff3fKSfsfS+ShIh0yf37TW48UsytJWMC1izIdDaq', region_name='ap-northeast-2')
    s3.download_file('donggyu-bucket', decoding, 'test_video.mp4')
    return 'download'

def posting_data(video_pk, emotion):
    url_items = "http://52.78.55.111/api/video/emotion"
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
        rnn_output_dim = 2 * hidden_dim if self.rnn.bidirectional else hidden_dim
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
    audio_model = UrbanSoundRNN(feat_dim=40, hidden_dim=64, num_class=10).to(device)
    audio_path = "model_cat_rnn_entire.pt"
    audio_model = torch.load(audio_path, map_location=torch.device('cpu'))
    print(audio_model.eval())
    return device, audio_model

def getaudio(file_name):
    sound, sample_rate = librosa.load(file_name, sr=16000, mono=True, res_type='kaiser_fast')
    melspec = librosa.feature.melspectrogram(y=sound, sr=sample_rate, n_mels=40)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    log_melspec = log_melspec.T
    sound_feature = torch.FloatTensor(log_melspec)
    target = 0
    return sound_feature, target

def video_inference(video='test_video.mp4'):
    device, model = setting_default()
    videoclip = mp.VideoFileClip(video)
    videoclip.audio.write_audiofile("test_audio.wav") 
    X, Y = getaudio('test_audio.wav')
    with torch.no_grad():
        seq_len, feat_dim = X.size()
        X = X.view(-1, seq_len, feat_dim).to(device)
        pred = model(X)
        print(pred)
        print(pred.argmax(1).item())
        wav_file_path = os.path.join('test_audio.wav')
        print(wav_file_path)
    emo_dict = {0: 'chattering', 1: 'growling', 2: 'hissing', 3: 'meowing', 4: 'purring', 5: 'trilling', 6: 'yelling', 7: 'noise'}
    return emo_dict[pred.argmax(1).item()]

def extracting_frame(video='test_video.mp4'):
    vidcap = cv2.VideoCapture(video)
    count = 0
    preds = []
 
    while(vidcap.isOpened()):
        ret, image = vidcap.read()
        if not ret:
            break
            
        if(int(vidcap.get(1)) % 30 == 0):
            PIL_image = Image.fromarray(image)
            pred = action_inference(PIL_image)
            preds.append(pred)
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            print('frame%d.jpg' % count)
            count += 1
    
    pred_most = max(preds, key=preds.count)
    vidcap.release()
    emo_dict = {0: 'chattering', 1: 'growling', 2: 'hissing', 3: 'meowing', 4: 'purring', 5: 'trilling', 6: 'yelling', 7: 'noise'}
    return emo_dict[pred_most.item()]

def action_inference(image, model_path='model_cat_mobilenetV2_statedict.pt'):
    if torch.cuda.is_available():
        is_cuda = True
    else:
        is_cuda = False
        
    model_ft = models.mobilenet_v2(pretrained=True)
    num_ftrs = model_ft.classifier[1].out_features
    model_ft.fc = nn.Linear(num_ftrs, 7)
    
    if is_cuda:
        model_ft = model_ft.cuda()
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    inputs = simple_transform(image)  # torch.Size([3, 224, 224])
    inputs = inputs.reshape(-1, 3, 224, 224)  # torch.Size([1, 3, 224, 224])

    if torch.cuda.is_available():
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)

    outputs = model_ft(inputs)
    _, preds = torch.max(outputs.data, 1)
    print('preds = ', preds)
    return preds


from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/api/video/upload', methods=['POST'])
def get_video():
    result = request.get_json()
    fileUrl, videoId = parsing_json(result)
    downloading_s3(fileUrl)
    audio_emotion = video_inference()
    action_emotion = extracting_frame()
    if audio_emotion == action_emotion or audio_emotion=='noise':
        emotion = action_emotion
    else:
        print(audio_emotion, action_emotion)
        emotion = audio_emotion
    posting_data(videoId, emotion)
    return 'success'

if not app.debug:
    import logging
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'dave_server.log', maxBytes=2000, backupCount=10)
    file_handler.setLevel(logging.WARNING)
    app.logger.addHandler(file_handler)

@app.errorhandler(404)
def page_not_found(error):
    app.logger.error('page_not_found')
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=False)
