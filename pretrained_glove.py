import bcolz
import pickle
import numpy as np
from config import *

def process_data(emb_dim, glove_path):
	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.{emb_dim}.dat', mode='w')

	cnt = 0

	with open(f'{glove_path}/glove.6B.{emb_dim}d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word2idx[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)
	        cnt += 1
	    
	print(cnt)
	vectors = bcolz.carray(vectors[1:].reshape((-1, emb_dim)), rootdir=f'{glove_path}/6B.{emb_dim}.dat', mode='w')
	vectors.flush()
	pickle.dump(words, open(f'{glove_path}/6B.{emb_dim}_words.pkl', 'wb'))
	pickle.dump(word2idx, open(f'{glove_path}/6B.{emb_dim}_idx.pkl', 'wb'))


def get_glove_embedding(emb_dim, glove_path):
	vectors = bcolz.open(f'{glove_path}/6B.{emb_dim}.dat')[:]
	words = pickle.load(open(f'{glove_path}/6B.{emb_dim}_words.pkl', 'rb'))
	word2idx = pickle.load(open(f'{glove_path}/6B.{emb_dim}_idx.pkl', 'rb'))

	glove = {w: vectors[word2idx[w]] for w in words}

	return glove

def create_embedding(emb_dim, target_vocab, glove_path):
	glove = get_glove_embedding(emb_dim, glove_path)
	
	matrix_len = len(target_vocab)
	weights_matrix = np.zeros((matrix_len, emb_dim))
	words_found = 0

	for i, word in enumerate(target_vocab):
	    try: 
	        weights_matrix[i] = glove[word]
	        words_found += 1
	    except KeyError:
	        # weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
	        weights_matrix[i] = np.zeros((emb_dim, ))

	return weights_matrix


# def main():
# 	# a = process_data(50, '/Users/divyanshumund/Downloads/glove/')
# 	b = get_glove_embedding(50, '/Users/divyanshumund/Downloads/glove/')
# 	print(b['the'])

if __name__== "__main__":
	process_data(args['embed_size'], args['glove_path'])