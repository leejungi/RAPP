import numpy as np 
import torch
import torch.nn as nn

def block(in_size, out_size, relu=0.01, act=True):
	if act == True:
		return nn.Sequential( nn.Linear(in_size, out_size),
							 nn.LeakyReLU(relu),
							 nn.BatchNorm1d(out_size)
			)
	else:
		return nn.Sequential( nn.Linear(in_size, out_size),
							 nn.BatchNorm1d(out_size)
			)

class AE(nn.Module):
	def __init__(self, in_size, relu=0.01, num_layer= 9, rep_dim = 20):
		super(AE, self).__init__()
		
		diff = (in_size - rep_dim)/(num_layer+1)
		hidden_size = [in_size] +[int(in_size-diff*(i+1)) for i in range(num_layer)] +[rep_dim]
		self.encoder_layers =[]
		for i in range(len(hidden_size)-2):		 
			self.encoder_layers += [block(hidden_size[i], hidden_size[i+1], relu=relu)]
		self.encoder_layers += [block(hidden_size[-2], hidden_size[-1], act=False)]
		self.encoder = nn.Sequential(*self.encoder_layers)
		
		self.decoder_layers =[]
		for i in reversed(range(2,len(hidden_size))): 
			self.decoder_layers += [block(hidden_size[i], hidden_size[i-1])]
		self.decoder_layers += [block(hidden_size[1], hidden_size[0], act=False)]
		self.decoder = nn.Sequential(*self.decoder_layers)
		
	def forward(self, x):		 
		return self.decoder(self.encoder(x))
