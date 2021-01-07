import io
import json
import torch
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from models import Net
from torch.autograd import Variable

app = Flask(__name__)

def transform_image(image_bytes):
	data_transforms = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor()
	])
	image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
	image = data_transforms(image).unsqueeze(0)
	image = Variable(image)
	return image

def get_prediction(image_bytes):
	model = Net()
	model.load_state_dict(torch.load('model.pt',map_location='cpu'),strict=False)
	model.eval()
	tensor = transform_image(image_bytes=image_bytes)
	outputs = F.softmax(model(tensor),dim=1)
	top_p,top_class = outputs.topk(1, dim = 1)
	return top_p,top_class
	
@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		file = request.files['file']
		img_bytes = file.read()
		bclass = ['Clear','Haze']
		prob,res = get_prediction(image_bytes=img_bytes)
		prob,res = prob.tolist(),res.tolist()
	return jsonify({'probability':float(prob[0][0]),'index':int(res[0][0]),'label':bclass[res[0][0]]})

if __name__ == '__main__':
    app.run(debug=False)