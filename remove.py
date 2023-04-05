from flask import Flask, render_template, request, send_file
import cv2
import tensorflow as tf
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from PIL import Image
import torch
import clip
import numpy as np
import os

app= Flask(__name__)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
checkpoint = torch.load("model_checkpoint/model_10.pt", map_location=torch.device('cpu'))


model.load_state_dict(checkpoint['model_state_dict'])

PATH= 'remove' 
@app.route('/', methods=['POST'])
def search():
    if request.method=='POST':
        content= request.json
        text= content['text']
        n= content['n']
        text_in = clip.tokenize([text]).to(device)
        images=[]
        list2=[]
        for i in os.listdir(PATH):
            for dr in os.listdir(os.path.join(PATH, i)):
                paths= os.path.join(PATH,i, dr)
                image = preprocess(Image.open(paths)).unsqueeze(0).to(device)
                list2.append(paths)
                images.append(image)

        image_in= torch.cat(images)
        outputs=[]
        with torch.no_grad():
    
            logits_per_image, logits_per_text = model(image_in, text_in)
            list1 = logits_per_text.softmax(dim=-1).cpu().numpy()
            
            result= sort_by_indexes(list2, list1[0])
            list1_sort= sorted(list1, reverse=True)
            
            for i in range(n):
                out= {"product_name": text,
                      "url": result[i],
                      "score": list1_sort[0][i]}
                outputs.append(out)
            return str(outputs)

 
def sort_by_indexes(lst, indexes, reverse=True):
  return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: x[0], reverse=reverse)]

if __name__=='__main__':
    app.run(debug=True)