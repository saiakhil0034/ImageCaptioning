from models.architectures import *
import torch
import torch.nn as nn
from config import arguments
import os
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print('arguments', arguments)
	if(not os.path.exists(arguments['model_path'])):
		os.makedirs(arguments['model_path'])

	encoder = arguments['encoder'](arguments['embed_size']).to(device)
	decoder = arguments['decoder'](arguments['embed_size'], arguments['hidden_size'], arguments['vocab_size'], arguments['num_layers']).to(device)

	criterion = arguments['loss_criterion']
	params = list(list(encoder.parameters()) + list(decoder.parameters()))
	#TODO: change optimizer to Adam
	optimizer = torch.optim.SGD(params, lr=arguments['learning_rate'])

	trainDataloader = get_loader(arguments['root'], arguments['train_json_path'], arguments['train_image_ids'],arguments['vocabulary'], arguments['transforms'], arguments['batch_size'], True, arguments['num_workers'])
	valDataloader = get_loader(arguments['root'], arguments['train_json_path'], arguments['val_image_ids'], arguments['vocabulary'], arguments['transforms'], arguments['batch_size'], True, arguments['num_workers'])

	trainLosses = []
	valLosses = []
	bestLoss = 2e10
	for epoch in range(arguments['epochs']):
		print("epoch", epoch)
		encoder.train()
		decoder.train()
		currentTrainLoss = []
		for idx, (images, captions, lengths) in enumerate(trainDataloader):

			images = images.to(device)
			captions = captions.to(device)
			targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

			imageFeatures = encoder(images)
			decoderOutputs = decoder(imageFeatures, captions, lengths)

			loss = criterion(decoderOutputs, targets)
			encoder.zero_grad()
			decoder.zero_grad()
			loss.backward()
			optimizer.step()

			currentTrainLoss.append(loss.item())
		trainLosses.append(avg(currentTrainLoss))

		if(epoch % arguments['val_step'] == 0):
			encoder.eval() 
			decoder.eval()
			currentValLoss = []
			for (images, captions, lengths) in valDataloader:

				images = images.to(device)
				captions = captions.to(device)
				targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

				imageFeatures = encoder(images)
				decoderOutputs = decoder(imageFeatures, captions, lengths)

				loss = criterion(decoderOutputs, targets) ##Need to figure out
				encoder.zero_grad()
				decoder.zero_grad()

				currentValLoss.append(loss.item())
			valLosses.append(avg(currentTrainLoss))
			if(valLosses[-1] < bestLoss):
				bestLoss = valLosses[-1]
				

main()





