import torch.nn as nn
import pickle
baseDataPath = './data/'
def gD(dir):
	return baseDataPath + dir

def get_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        str_ids = file.read().split(",")
        ids = [int(_ids) for _ids in str_ids]
    return ids

# baseResultPath = './results/'
# def gR(dir):
# 	return baseResultPath + fileName

from torchvision import transforms

size = (224,224) # refer to data_loader_captions notebook
transforms_ = transforms.Compose([
					transforms.Resize(256),
                    transforms.CenterCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_ids = get_ids('./data/train_ids.txt')
val_ids = get_ids('./data/val_ids.txt')
test_ids = get_ids('./data/test_ids.txt')

from models.architectures import EncoderCNN, DecoderLSTM, DecoderRNN

vocab1 = ''
vocab_path1 = './vocab/word_to_idx.p'
with open(vocab_path1, 'rb') as f:
	vocab1 = pickle.load(f)

vocab2 = ''
vocab_path2 = './vocab/idx_to_word.p'
with open(vocab_path2, 'rb') as f:
	vocab2 = pickle.load(f)

args = {

	'epochs' : 1000,

	'batch_size' : 64,

	'num_workers' : 32,

	'model_path' : './results/mega6',

	'embed_size' : 300,

	'hidden_size' : 750,

	'val_step' : 3,

	'encoder' : EncoderCNN,

	'decoder' : DecoderRNN,

	'vocabulary' : vocab1,
    
    'vocabulary_' : vocab2,

	'vocab_size' : len(vocab1),

	'num_layers' : 1,

	'loss_criterion' : nn.CrossEntropyLoss(),

	'learning_rate' : 1e-4,
    
    'beta' : 0.9,

	'train_image_ids' : train_ids,

	'train_json_path' : gD('annotations/captions_train2014.json'),

	'val_image_ids' : val_ids,

	'val_json_path' : gD('annotations/captions_train2014.json'),

	'test_image_ids' : test_ids,

	'test_json_path' : gD('annotations/captions_val2014.json'),

	'transforms' : transforms_,

	'root' : '/datasets/COCO-2015/train2014/',
    
    'root_' : '/datasets/COCO-2015/val2014/',

    'generate_mode' : 'deterministic',

    'temperature' : 3,

    'glove_path' : './data/glove',

    'max_sentence_length' : 100,

    'is_pretrained' : False

}