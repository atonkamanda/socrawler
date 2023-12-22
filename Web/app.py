from flask import Flask, render_template, request, url_for
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch
import torch.nn as nn
import numpy as np
import gc

# Refer to print to understand what's going on. 
app = Flask(__name__)
print("Please wait the server need to proceed few steps ....")
print("Loading dataset ... 1/4")
df = pd.read_csv('codegit.csv')
print("Loading model ... 2/4")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("./python_model")
print("Checking for GPU capability ...3/4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Loading vector_space ... 4/4")
vector_space = torch.load('vector_space_234853.pt')
print("Done !")


@app.route('/')
def main_page():
    return render_template("main_page.html")

@app.route('/results', methods=['GET'])
def results_page():
	query = request.args.get('query')
	result_to_display = 10
	code_list = []
	git_list = []
	with torch.no_grad():
		query_vec = model(tokenizer(query,return_tensors='pt')['input_ids'].cuda())[1]
		scores=torch.einsum("ab,cb->ac",query_vec.cuda(),vector_space.cuda())
		scores=torch.softmax(scores,-1)
		score_list ,indices = scores.topk(result_to_display)
		score_list = score_list.tolist()[0]
		indices = indices.tolist()[0]
	for i in range(result_to_display):
		code_list.append(df["code"][indices[i]])
		git_list.append(df["url"][indices[i]])
		

	return render_template("results_page.html",data=query, result = result_to_display, codes=code_list,dist=score_list,git=git_list)

if __name__ == "__main__":
    app.run()

