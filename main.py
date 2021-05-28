import numpy as np
import torch
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from argparse import RawTextHelpFormatter
import time
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor 
import torch.nn as nn
import matplotlib.pyplot  as plt
from datasets import Dataset
from model import AE
from sklearn.metrics import roc_auc_score, roc_curve, auc


parser = argparse.ArgumentParser(description='RAPP')
#Model params
parser.add_argument('--device', type=str, default='cuda', help="Device name for torch executing environment e.g) cpu, cuda")
parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
parser.add_argument('--epochs', type=int, default=200, help="AutoEncoder Num of Epochs")
parser.add_argument('--learning_rate', type=float, default=1e-3, help="AutoEncoder Learning Rate")
parser.add_argument('--weight_decay', type=float, default=0, help="AutoEncoder Weight decay in optimizer")

parser.add_argument('--normal_class', type=list, default=[0,2,3,4,5,6,7,8,9], help="AutoEncoder Weight decay in optimizer")

parser.add_argument('--test_interval', type=int, default=10, help="Test and model save interval during train")
parser.add_argument('--tensorboard', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')

parser.add_argument('--train', type=int, default=1, help="Training start Flag")
parser.add_argument('--test', type=int, default=1, help="Test start Flag")
parser.add_argument('--load_model', type=str, default="model.pth", help="Load model name")
args = parser.parse_args()

def get_layer_diff(X, recon, model):
	model.eval()
	with torch.no_grad():
		diff = [X-recon]
		for layer in model.encoder_layers[:-1]:
			X = layer(X)
			recon = layer(recon)
			diff.append(X-recon)
	diff = torch.cat(diff[1:], 1)
	return [diff]
#	 return [torch.mean(sap_diff**2,1)], [torch.mean(rsap_diff**2,1)]

def Snap(recon, mu, s, v):
	center_recon = recon-mu
#	inv_s = torch.inverse(torch.diag(s))
#	nap = torch.mean(torch.mm(torch.mm(center_recon,v),inv_s)**2, 1)
	DV = torch.mm(center_recon,v)
#	DV_mu = DV.mean(0)
#	DV_var = DV.var(0)
#	nap = torch.mean((DV-DV_mu)/DV_var,1)
	nap = torch.mean((DV/s)**2, 1)
	return nap.to('cpu')

def set_svd(recon):
	mu = recon.mean(0)
	center_recon = recon-mu
	_, s, v = center_recon.svd()
	return mu, s, v

def TEST(model, train_loader, test_loader, device, epoch=None,writer=None, valid=False):
	
	recon = []
	Sord_list = []
	label = []
	
	model.eval()
	with torch.no_grad():
		for X, _ in train_loader:
			X = X.to(device)

			hypothesis = model(X)

			recon += get_layer_diff(X,hypothesis,model)

		recon= torch.cat(recon).to('cpu')
		mu, s, v =set_svd(recon)

		recon =[]
		for X, Y in test_loader:
			X = X.to(device)
			Y = Y.to(device)
			hypothesis = model(X)
			Sord_list += [torch.mean((hypothesis - X)**2,1)]
			recon += get_layer_diff(X,hypothesis,model)
			label += [Y]
	recon= torch.cat(recon).to('cpu')

	Sord_list= torch.cat(Sord_list).to('cpu')

#	Ssap_list = torch.cat(recon).to('cpu')
	Ssap_list = torch.mean(recon**2,1)

	Snap_list = Snap(recon, mu, s, v)

	label = torch.cat(label).to('cpu')
	Sord_auroc=roc_auc_score(label,Sord_list)
	Ssap_auroc=roc_auc_score(label,Ssap_list)
	Snap_auroc=roc_auc_score(label,Snap_list)
	print("Test Sord auroc: {:.3f} Ssap auroc: {:.3f} Snap auroc: {:.3f}".format(Sord_auroc, Ssap_auroc, Snap_auroc))
	if epoch != None:
		writer.add_scalar('Valid/Sord AUROC score', round(Sord_auroc,3), epoch)
		writer.add_scalar('Valid/Ssap AUROC score', round(Ssap_auroc,3), epoch)
		writer.add_scalar('Valid/Snap AUROC score', round(Snap_auroc,3), epoch)
	if valid==True:
		hypothesis = hypothesis.view(-1,1,28,28)
		img_grid = torchvision.utils.make_grid(hypothesis)
		writer.add_image('Test/AE_mnist_images', img_grid,0)
		
		X = X.view(-1,1,28,28)
		img_grid = torchvision.utils.make_grid(X)
		writer.add_image('Test/mnist_images', img_grid,0)
		

def main():
	# Create a SummaryWriter object by TensorBoard
	if args.tensorboard:
		dir_name = 'tf_board/' + '_s_' + str(args.seed) \
						   + '_t_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		writer = SummaryWriter(log_dir=dir_name)
		
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	
	if args.device == 'cuda':
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	else:
		device = torch.device('cpu')
		
	if device == 'cuda':		
		torch.cuda.manual_seed_all(args.seed)
		
	train_data = MNIST(root='./', train=True, download=True, transform=ToTensor())
	test_data = MNIST(root='./', train=False, download=True, transform=ToTensor())
		 
	train_dset = Dataset(train_data, test_data, normal=args.normal_class, train=True)
	test_dset = Dataset(train_data, test_data, normal=args.normal_class, train=False)
	
	train_loader = torch.utils.data.DataLoader(dataset=train_dset,
											  batch_size=args.batch_size,
											  shuffle=True,
											  drop_last=False)
	test_loader = torch.utils.data.DataLoader(dataset=test_dset,
											  batch_size=args.batch_size,
											  shuffle=False,
											  drop_last=False)
	
	model = AE(28*28).to(device)
	
	criterion = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
		
	if args.train == True:
		total_batch = len(train_loader)
		for epoch in range(args.epochs):
			avg_cost=0
			model.train()
			#Training
			for X, _ in train_loader:
				X = X.to(device)
				
				optimizer.zero_grad()
				hypothesis = model(X)
				
				cost = criterion(hypothesis, X)
				cost.backward()#Gradient calculation
				optimizer.step()#Gradient update
				
				avg_cost += cost/(total_batch*args.batch_size)

			avg_cost = float(avg_cost.to('cpu').detach())

			
			
			# Log and test
			if (epoch+1)%args.test_interval ==0:
				torch.save(model.state_dict(),'save_model/AE_'+str(epoch+1)+'.pth')
				writer.add_scalar('Train/Avg Loss', round(avg_cost,7), epoch)
				print("{}/{} Train Avg Loss: {}".format(epoch+1,args.epochs,avg_cost))
				
				#Test
				TEST(model, train_loader, test_loader, device, epoch=epoch, writer=writer, valid=True)
		
		torch.save(model.state_dict(),'save_model/model.pth')
		
		hypothesis = hypothesis.view(-1,1,28,28)
		img_grid = torchvision.utils.make_grid(hypothesis)
		writer.add_image('Training/AE_mnist_images', img_grid,0)
		
		X = X.view(-1,1,28,28)
		img_grid = torchvision.utils.make_grid(X)
		writer.add_image('Training/mnist_images', img_grid,0)
		
	if args.test == True:  
		model.load_state_dict(torch.load('save_model/'+str(args.load_model)))
		TEST(model, train_loader, test_loader, device,writer=writer)
		

					
if __name__ == "__main__":
	main()
