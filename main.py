
from flask import Flask, render_template, request , jsonify
# from werkzeug.utils import html
import pickle
#from sklearn.linear_model import LogisticRegression
import re 
from prediction import predict
import numpy as np
#import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index_m.html')

@app.route('/dna', methods=['GET', 'POST'])
def species_identification():
	if request.method == 'POST':
		def encode_sequence(sequence):
			encoding = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
			return [encoding[base] for base in sequence]
		with open('DNA based Recognition/species.pkl', 'rb') as f:
			classifier = pickle.load(f)
		label_encoder = LabelEncoder()
		label_encoder.classes_ = np.load('DNA based Recognition/label_encoder.npy',allow_pickle=True)
		form_data = request.form
		dna = form_data.get('DNA Sequence').upper()
		print(dna)
		str_dna = re.sub('[^ATCG]', '', dna)
		encoded_str = encode_sequence(str_dna)
		lst=[encoded_str]
		print(lst)
		str_padded =pad_sequences(lst, maxlen=1128, padding='post')
		print(str_padded)
		predicted_probs = classifier.predict_proba(str_padded)
		print("2")
		top_n = 2
		ans=[]
		top_3_category = np.argsort(predicted_probs)[:, (- top_n-1) :]
		for i in top_3_category[0]:
			decode=  label_encoder.inverse_transform([i])
			ans.append(decode)
		top_3_probs = np.sort(predicted_probs)[:,( - top_n-1) :]
		rev_name=list(reversed(ans))
		rev_prob =list(reversed(top_3_probs[0]))
		for i in range(len(rev_prob)):
			rev_prob[i]=round(rev_prob[i],3)
		
		print(rev_name)
		print(rev_prob)

	return render_template('index_m.html',data=zip(rev_name,rev_prob))

@app.route('/image', methods=['GET', 'POST'])
def image_identification():
	if request.method == 'POST':
		# if 'img' not in request.files:
		# 	return "No image part"   
		img = request.files.get('img','')
		if img == '':
			return "No selected image"
		# img= request.files.get('img', '')
		print(img)
		prob,category =predict(img,'Image based Recognition/best_model.pth','Image based Recognition/picture_identify_label_encoder.npy')
		print(category)
		print(prob)
	return render_template('index_m.html',data2=zip(category,prob))

if __name__ == '__main__':
	 app.run(host="localhost", port=8800, debug = True)
