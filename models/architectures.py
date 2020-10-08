import torch
import torch.nn as nn
import torchvision.models as pretrained
import numpy as np
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from pretrained_glove import create_embedding
import sys


#resNetComplete = pretrained.resnext50_32x4d(pretrained=True)

class EncoderCNN(nn.Module):
	def __init__(self, output_size):
		super(EncoderCNN, self).__init__()
		resNetComplete = pretrained.resnet50(pretrained=True)
		subModules = list(resNetComplete.children())[:-1]
		self.resNetToUse = nn.Sequential(*subModules)
		for param in resNetComplete.parameters():
			param.requires_grad = False
		self.lastLayer = nn.Linear(resNetComplete.fc.in_features, output_size)
		self.batchNorm = nn.BatchNorm1d(output_size)

	def forward(self, image_inputs):
		imageFeatures = self.resNetToUse(image_inputs)
		imageFeatures = imageFeatures.reshape(imageFeatures.size(0), -1)
		outputFeatures = self.batchNorm(self.lastLayer(imageFeatures))
		return outputFeatures

class DecoderLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, vocab_size, num_layers, target_vocab, glove_path, max_sentence_length = 100, is_pretrained = False):
		super(DecoderLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.outputLayer = nn.Linear(hidden_size, vocab_size)
		self.embeddingLayer = nn.Embedding(vocab_size, input_size)

		if(is_pretrained):
			weights_matrix = create_embedding(input_size, target_vocab, glove_path)
			self.embeddingLayer = nn.Embedding.from_pretrained(torch.Tensor(weights_matrix))

		self.maxSentenceLength = max_sentence_length

	def forward(self, encoder_outputs, captions, lengths):
		wordEmbeddings = self.embeddingLayer(captions)
		wordEmbeddings = torch.cat((encoder_outputs.unsqueeze(1), wordEmbeddings), 1)
		hiddenStates, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(wordEmbeddings, lengths, batch_first=True))
		#print(hiddenStates[0].shape)
		vocabScores = self.outputLayer(hiddenStates[0])
		return vocabScores

	def forwardEval(self, features, states=None, mode='deterministic',t=1):
		features = features.unsqueeze(1)
		all_words = []
		all_outputs = []
		for i in range(self.maxSentenceLength):
			hiddens, states = self.lstm(features, states)
			curOutputs = self.outputLayer(hiddens.squeeze(1))
			#print('cur outputs : ', curOutputs.size())
			all_outputs.append(curOutputs)
			if(mode == 'stochastic'):
				soft_out = F.softmax(curOutputs/t, dim=1)
				i_words = WeightedRandomSampler(torch.squeeze(soft_out), 1)
				i_words = torch.multinomial(soft_out, 1)
			else:
				_,predicted = curOutputs.max(1)
				i_words = predicted.unsqueeze(1)
			#print(i_words.size())
			inputs = self.embeddingLayer(i_words)
			features = inputs
			all_words.append(i_words)
		all_words = torch.stack(all_words, 1)
		all_outputs = torch.stack(all_outputs, 1)
		return all_words, all_outputs

class DecoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, vocab_size, num_layers, target_vocab, glove_path, max_sentence_length = 100, is_pretrained = False):
		super(DecoderRNN, self).__init__()
		self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
		self.outputLayer = nn.Linear(hidden_size, vocab_size)
		self.embeddingLayer = nn.Embedding(vocab_size, input_size)

		if(is_pretrained):
			weights_matrix = create_embedding(input_size, target_vocab, glove_path)
			self.embeddingLayer = nn.Embedding.from_pretrained(torch.Tensor(weights_matrix))
			# self.embeddingLayer.load_state_dict({'weight' : weights_matrix})
			# self.embeddingLayer.weight.requires_grad = False

		self.maxSentenceLength = max_sentence_length

	def forward(self, encoder_outputs, captions, lengths):
		wordEmbeddings = self.embeddingLayer(captions)
		wordEmbeddings = torch.cat((encoder_outputs.unsqueeze(1), wordEmbeddings), 1)
		hiddenStates, _ = self.rnn(torch.nn.utils.rnn.pack_padded_sequence(wordEmbeddings, lengths, batch_first=True))
		#print(hiddenStates[0].shape)
		vocabScores = self.outputLayer(hiddenStates[0])
		return vocabScores

	def forwardEval(self, features, states=None, mode='deterministic',t=1):
		features = features.unsqueeze(1)
		all_words = []
		all_outputs = []
		for i in range(self.maxSentenceLength):
			hiddens, states = self.rnn(features, states)
			curOutputs = self.outputLayer(hiddens.squeeze(1))
			#print('cur outputs : ', curOutputs.size())
			all_outputs.append(curOutputs)
			if(mode == 'stochastic'):
				soft_out = F.softmax(curOutputs/t, dim=1)
				i_words = WeightedRandomSampler(torch.squeeze(soft_out), 1)
				i_words = torch.multinomial(soft_out, 1)
			else:
				_,predicted = curOutputs.max(1)
				i_words = predicted.unsqueeze(1)
			#print(i_words.size())
			inputs = self.embeddingLayer(i_words)
			features = inputs
			all_words.append(i_words)
		all_words = torch.stack(all_words, 1)
		all_outputs = torch.stack(all_outputs, 1)
		#print('final outputs : ', all_outputs.size())
		return all_words, all_outputs




	# def generate(self, features, states=None):
	# 	word_ids = []
	# 	features = features.unsqueeze(1)
	# 	for i in range(self.maxSentenceLength):
	# 		hiddens, states = self.lstm(features, states)
	# 		outputs = self.outputLayer(hiddens.squeeze(1))
	# 		_ , predicted = outputs.max(1)
	# 		word_ids.append(predicted)
	# 		inputs = self.embeddingLayer(predicted)
	# 		inputs = inputs.unsqueeze(1)
	# 	word_ids = torch.stack(word_ids, 1)
	# 	return word_ids
    
	# def generate_captions(self, logits, mode='deterministic', t=1):
	# 	if (mode == 'deterministic'):
	# 		_, predicted = logits.max(1)
	# 		word_id = predicted
        
	# 	elif(mode == 'stochastic'):
	# 		soft_out = F.softmax(logits/t, dim=1)
	# 		word_id = WeightedRandomSampler(torch.squeeze(soft_out), 1) #get only one sample. change it to get more samples
        
	# 	return word_id
            
            
            
            
            
        

# e = EncoderCNN(300)
# output = e(torch.zeros(3, 3, 224, 224))
# d = DecoderLSTM(300, 200, 50, 1, 5)
# o = d(output, torch.from_numpy(np.array([[1,2,1,3,3],[1,2,1,3,3],[1,2,1,3,3]])), torch.from_numpy(np.array([3,2,1])))








