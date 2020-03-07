import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt

def load_doc():
	documents = []
	art = []
	path1 = "~/art/*"
	path2 = "~/news/"
	path1 = glob.glob(path1)

	for i in range(1, 201):
		file = open(path2 + str(i), "r").read()
		temp = np.asarray(file.split(" "))
		temp = list(filter(None, temp))
		documents.append(temp)

	for f in path1:
		file = open(f, "r").read()
		temp = np.asarray(file.split(" "))
		temp = list(filter(None, temp))
		art.append(temp)

	documents = np.asarray(documents)
	art = np.asarray(art)

	return art, documents


def vocab_count(documents, w_n, words):
	vocab = {}
	count = 0
	num_words = 0
	v = []

	for d in documents:
		for w in d:
			if w not in vocab:
				vocab[w] = count
				count += 1
			num_words += 1
	for n in range(num_words):
		w_n[n] = vocab[words[n]]

	for w in vocab:
		v.append(w)

	return w_n, count, num_words, v


def permute_docs(w_n):
	pi = np.array([i for i in range(len(w_n))])
	np.random.shuffle(pi)
	return pi


def concat_docs(documents):
	words = []
	d_n = []
	i = 0
	for d in documents:
		for w in d:
			d_n.append(i)
		words.extend(d)
		i += 1
	words = list(filter(None, words))
	w_n = np.zeros(len(words))

	return w_n, d_n, words


def init_C_d(K, z_n, documents):
	C_d = np.zeros((documents.shape[0], K))
	temp = np.zeros(K)
	z = 0
	d = 0

	for doc in documents:
		for w in doc:
			temp[z_n[z]] += 1
			z += 1
		for i in range(K):
			C_d[d, i] = temp[i]
		d += 1


	return C_d


def init_C_t(K, V, z_n, w_n):
	C_t = np.zeros((K, V))

	for w in range(len(w_n)):
		C_t[z_n[w], int(w_n[w])] += 1

	return C_t


def calc_P_k(P, K, beta, word, doc, alpha, C_d, C_t, V):
	for k in range(K):
		top_left = C_t[k][word] + beta
		temp = np.sum(C_t[k, :])
		bot_left = V * beta + temp

		top_right = C_d[doc][k] + alpha

		temp = np.sum(C_d[doc, :])

		bot_right = K * alpha + temp
		P[k] = (top_left / bot_left) * (top_right / bot_right)

	return P


def normalize(P):
	total = sum(P)
	for k in range(len(P)):
		P[k] = P[k] / total

	return P


def sample(P, K):
	return np.random.choice(np.arange(0, K), p=P)


def most_freq_w(C_t, vocab, K):
	top_words = []
	for k in range(K):
		temp = C_t[k, :]
		max = 0
		indices = []
		index = 0
		for i in range(5):
			for j in range(len(temp)):
				if temp[j] > max:
					max = temp[j]
					index = j
			indices.append(index)
			temp[index] = 0
			max = 0
		topic_w = []
		for i in indices:
			topic_w.append(vocab[i])
		top_words.append(topic_w)

	return top_words


def topic_representation(C_d, K, alpha):
	doc_reps = []

	for d in range(C_d.shape[0]):
		top_rep = np.zeros(K)
		for k in range(K):
			top_rep[k] = (C_d[d, k] + alpha) / (K * alpha + sum(C_d[d, :]))
		doc_reps.append(top_rep)

	return doc_reps


def task1(documents, alpha=.1, beta=.01, iters=100, K=20):
	w_n, d_n, words = concat_docs(documents)
	w_n, V, num_words, v = vocab_count(documents, w_n, words)
	z_n = np.random.randint(0, K, num_words)
	C_d = init_C_d(K, z_n, documents)

	pi = permute_docs(w_n)
	C_t = init_C_t(K, V, z_n, w_n)
	P = np.zeros(K)

	for i in range(iters):
		for n in range(num_words):
			word = int(w_n[pi[n]])
			topic = z_n[pi[n]]
			doc = d_n[pi[n]]
			C_d[doc, topic] = C_d[doc, topic] - 1
			C_t[topic, word] = C_t[topic, word] - 1
			P = calc_P_k(P, K, beta, word, doc, alpha, C_d, C_t, V)
			P = normalize(P)
			topic = sample(P, K)
			z_n[pi[n]] = topic
			C_d[doc, topic] = C_d[doc, topic] + 1
			C_t[topic, word] = C_t[topic, word] + 1

	freq_words = most_freq_w(C_t, v, K)
	print(freq_words)
	string = ""
	for l in freq_words:
		string += ' '.join(l) + "\n"

	f = open("topicwords.csv", "w")
	f.write(string)
	f.close()

	topic_rep = topic_representation(C_d, K, alpha)

	return z_n, C_d, C_t, topic_rep, words


def build_the_bag(doc, words, train):

	vocab = {}
	count = 0

	for d in doc:
		for w in d:
			if w not in vocab:
				vocab[w] = count
				count += 1

	doc_bag = []
	for d in doc:
		w_n = np.zeros(len(vocab))
		for w in d:
			w_n[vocab[w]] +=1
		doc_bag.append(w_n)

	# an array with the number of words in the vocabulary
	# for each document, for each word, use the vocab[w] to increment the w_n index

	for b in doc_bag:
		i = 0
		for w in b:
			b[i] = (b[i] /len(b))
			i += 1

	return np.asarray(doc_bag)


def train_test_split(doc, y):
	part = int(len(doc) * .6)
	train = doc[:part]
	test = doc[part:]
	y_train = y[:part]
	y_test = y[part:]

	return train, test, y_train, y_test

def predict(pred):
	preds = []
	for y in pred:
		if y > 0.5:
			preds.append(1)
		else:
			preds.append(0)
	return preds


def accuracy(y, pred):
	total = 0
	for i in range(len(y)):
		if y[i] == pred[i]:
			total += 1

	return float(total / (len(y)))

def sig(x):
	return 1 / (1 + np.exp(-x))

def check_stop(w_hat, w):
	return np.linalg.norm(w_hat - w) / np.linalg.norm(w)

def blr(X_train, X_test, y_train, alpha=1):
	y = y_train.reshape(-1, 1)
	w = np.zeros((len(X_train[0]), 1))

	w, R = Newton_Raphson(X_train, alpha, w, y)

	S_N = np.linalg.inv(((np.dot(np.dot(X_train.T, R), X_train)) + alpha * np.eye(len(X_train[0]))))
	post =[]
	for i in range(len(X_test)):
		x = X_test[i].reshape(1, -1)
		mu = np.dot(x, w)
		sig_a = np.dot(np.dot(X_test[i], S_N), X_test[i].T)
		p = sig(mu / np.sqrt(1.0 + (np.pi / 8 * sig_a)))
		post.append(p)

	return post

def Newton_Raphson(X_train, alpha, w, y):

	for i in range(100):
		R = np.zeros((len(X_train), len(X_train)))
		for j in range(len(X_train)):
			x = X_train[j].reshape(-1, 1)
			R[j][j] = (sig(np.dot(x.T, w)) * (1 - sig(np.dot(x.T, w)))).flatten()[0]
		sigma_ = sig(np.dot(X_train, w))
		S_N = np.linalg.inv(((np.dot(np.dot(X_train.T, R), X_train)) + (alpha + 1e-9) * np.identity(len(X_train[0]))))
		w_hat = w - np.dot(S_N, (np.dot(X_train.T, (sigma_ - y)) + alpha * w))

		if check_stop(w_hat, w) < 1e-3:
			break
		return w_hat, R

def calc_R(sig, shape):

	R = np.zeros((shape, shape), np.float32)
	for i in range(len(sig)):
		R[i,i] = (sig[i]) * (1 - sig[i])

	return R

def optimal_a(X, y, label):

	X = np.c_[X, np.ones(X.shape[0])]
	N = len(y)

	train_size = int(X.shape[0] * .6)
	x_train, x_test = X[:train_size, :], X[train_size:, :]
	y_train, y_test = y[:train_size], y[train_size:]
	size = np.arange(float(N) / 10, train_size, 10, dtype=np.int32) + 1
	acc_list = []
	alphas = []
	stddev_list = []
	for s in size:
		ave_acc = []
		ave_a = []
		for i in range(30):

			sample_x = x_train[:s, :]
			sample_y = y_train[:s]

			alpha = 1
			w = np.zeros((len(x_train[0]), 1))
			for l in range(10):
				w_map, R = Newton_Raphson(x_train, alpha, w, y)
				eig = np.linalg.eigh(np.dot(np.dot(x_train.T, R), x_train))[0]
				gamma = 0
				for v in eig:
					gamma += (v / (v + alpha))
				alpha = gamma / np.linalg.norm(w_map) ** 2

			post = blr(sample_x, x_test, sample_y, alpha=alpha)
			preds = predict(post)
			acc = accuracy(y_test, preds)
			ave_acc.append(acc)
			ave_a.append(alpha)

		stddev = np.std(ave_acc)
		stddev_list.append(stddev)
		ave_acc = np.mean(ave_acc)
		ave_a = np.mean(ave_a)
		alphas.append(ave_a)
		acc_list.append(ave_acc)


	plt.figure()
	plt.plot(size, acc_list, label='Accuracy')
	u = np.asarray(acc_list) + np.asarray(stddev_list)
	l = np.asarray(acc_list) - np.asarray(stddev_list)
	plt.fill_between(len(size), l, u, facecolor = "red")
	plt.xlabel('Training Size')
	plt.ylabel('Accuracy')

	plt.legend()
	plt.savefig(label + '.png', dpi=1000)


# Driver
art, d = load_doc()
# z_n, C_d, C_t, topic_rep, words = task1(art, K=2)

y = []
with open("index.csv", "r") as fp:
	for line in fp:
		y.extend(line.strip().split(','))
temp = []
for i in range(len(y)):
	if i%2 != 0:
		temp.append(float(y[i]))

y = np.asarray(temp)

z_n, C_d, C_t, topic_rep, words = task1(d)

with open("topic_rep.pickle", 'wb') as f:
	pickle.dump(topic_rep, f)

with open("words.pickle", 'wb') as f: 
	pickle.dump(words, f)

with open("topic_rep.pickle", 'rb') as f:
	topic_rep = np.asarray(pickle.load(f))

with open("words.pickle", 'rb') as f:
	words = pickle.load(f)

train, test, y_train, y_test = train_test_split(d, y)
bag = np.asarray(build_the_bag(d, words, train))

optimal_a(bag,y,"bow")
optimal_a(topic_rep,y, "lda")

