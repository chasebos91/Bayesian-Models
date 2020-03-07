
import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


def load_data():
	feats = ["A.csv", "B.csv", "crashes.csv", "diabetes.csv", "ionosphere.csv"]
	labels = [ "labels-A.csv", "labels-B.csv", "labels-crashes.csv", "labels-diabetes.csv", "labels-ionosphere.csv"]
	log_reg_ref = ["irlsw.csv",  "irlstest.csv", "labels-irlstest.csv"]

	data = []
	log_reg_data = []
	for i in range(len(feats)):
		x = np.genfromtxt(feats[i], delimiter=',')
		y = np.genfromtxt(labels[i], delimiter=',')
		data.append((x,y))

	for i in range(len(log_reg_ref)):
		d = np.genfromtxt(log_reg_ref[i], delimiter=',')
		log_reg_data.append(d)

	return data, log_reg_data

def task_1(data,lrd, a, task_title):

	#print(Bayes_LogReg_train(lrd[1], lrd[2], 1))
	a_val = 0
	mean_curves = []

	if task_title == "Task 3":
		data = data[1:]
		a_val = 1

	for d in data:
		blearning_curves = []
		blearning_curves2 = []
		glearning_curves = []
		gauss_curves = []

		N = d[1].shape[0]
		top = int(N * .6)
		M = int(top * .1)

		for i in range(30):
			start = 0
			feat, labels = shuffle_data(d[0], d[1])
			labels = labels.flatten()

			x_train = feat[:top, :]
			y_train = labels[:top]
			x_test = feat[top:, :]
			y_test = labels[top:]
			blc = []
			blc2 = []
			glc = []
			gaussc = []
			m = M

			while m < top:

				x = x_train[start:m, :]
				y = y_train[start:m]

				if data_check(y) == True:

					feat, labels = shuffle_data(d[0], d[1])
					labels = labels.flatten()
					x_train = feat[:top, :]
					y_train = labels[:top]
					x_test = feat[top:, :]
					y_test = labels[top:]

				else:

					for val in range(len(a[a_val])):
						wmap = Bayes_LogReg_train(x, y, a[a_val][val])
						err = 1 - Bayes_LogReg_Test(wmap, a[a_val][val], x_test, y_test)
						if val ==0:
							blc.append(err)
						else: blc2.append(err)

					posterior = Generative_model(x, y, x_test)
					err = 1 - accuracy_generative(posterior, y_test)
					glc.append(err)

					if task_title == "Task 3":
						err = task_3(x, y, x_test, y_test)
						gaussc.append(err)

					m += M

			blearning_curves.append(blc)
			blearning_curves2.append(blc2)
			glearning_curves.append(glc)
			gauss_curves.append(gaussc)

		bmean = np.mean(blearning_curves, axis=0)
		bmean2 = np.mean(blearning_curves2, axis=0)
		gmean = np.mean(glearning_curves, axis=0)
		gaussmean = np.mean(gauss_curves, axis=0)
		mean_curves.append((bmean, gmean, bmean2, gaussmean))
		a_val += 1

	plot(mean_curves, task_title)

def plot(learning_curves, task):

	labels = ["Bayesian Logistic Regression", "Generative Model", "Bayesian Logistic with Regularization", "Gaussian Model"]
	titles = ["A", "B", "crashes", "diabetes", "ionosphere"]
	t = 0

	if task == "Task 3":
		titles = titles[1:]

	for lc in learning_curves:
		fig = plt.figure()
		plt.title(titles[t], fontsize=12, fontweight='bold')
		plt.suptitle(task, verticalalignment = 'bottom',fontsize=10)
		t += 1
		for i in range(len(lc)):
			if len(lc[i]) != 0:
				plt.plot(lc[i], label = labels[i])
		plt.legend(loc='upper right')
		plt.show()
		fig.savefig(task + str(t))

def data_check(y):

	for i in y:
		if i == 0: return False
	return True

def shuffle_data(x,y):
	x_i = x.shape[1]
	data = np.c_[x,y]
	np.random.shuffle(data)
	return data[:, :x_i], data[:,x_i:]

def Generative_model(x,y, xt):

	cov = np.cov(x.T)
	mu1, mu2 = calc_mu(x, y)
	pc1, pc2 = calc_pc(y)

	w = (np.linalg.pinv(cov).dot(mu1 - mu2))
	w0 = -(1/2 * mu1.T.dot(np.linalg.pinv(cov)).dot(mu1)) + 1/2 * mu2.T.dot(np.linalg.pinv(cov)).dot(mu2) + np.log((pc1/pc2))
	posterior = sigmoid(w.dot(xt.T) + w0)
	return posterior

def accuracy_generative(post, y_true):
	total = 0.0
	for i in range(len(post)):
		if post[i] > .5: post[i] = 1
		else: post[i] = 0
	for j in range(len(post)):
		if post[j] == y_true[j]:
			total += 1
	return total / len(post)

def calc_pc(y):
	pc1 = 0.0

	for i in y:
		if i == 1:
			pc1 += 1
	return pc1, len(y) - pc1

def calc_mu(x, y):
	x1, x2 = [], []
	for i in range(len(y)):
		if y[i] == 1:
			x1.append(x[i, :])
		else: x2.append(x[i, :])

	mu1 = np.asarray(x1).mean(axis=0)
	mu2 = np.asarray(x2).mean(axis=0)

	return mu1, mu2

def Bayes_LogReg_train(x,y,a):

	x = np.c_[x,np.ones(x.shape[0])]
	w = np.zeros((x.shape[1]))
	k = 1000

	w_map, k = Newton_raphson(x, y, w, k, a)

	return w_map

def Bayes_LogReg_Test(w_map, a, xt, yt):
	xt = np.c_[xt, np.ones(xt.shape[0])]
	predictive_dist = pred_dist(w_map, xt, a)

	preds = blr_predict(predictive_dist)
	acc = accuracy(preds, yt)
	return acc

def blr_predict(pred):
	preds = []

	for inst in range(len(pred)):

		if pred[inst] > .5: preds.append(1)
		else: preds.append(0)
	return preds

def accuracy(preds, y_true):
	total = 0
	for i in range(len(preds)):
		if preds[i] == y_true[i]:
			total +=1
	return total / len(y_true)

def sigmoid(a):

	sig = []
	for i in range(len(a)):
		sig.append(1 / (1 + np.exp(-a[i])))

	return sig

def calc_R(sig, shape):

	R = np.zeros((shape, shape), np.float32)
	for i in range(len(sig)):
		R[i,i] = (sig[i]) * (1 - sig[i])

	return R

def check_stop(w_hat, w):
	return np.linalg.norm(w_hat - w) / np.linalg.norm(w)

def Newton_raphson(x, y, w, k, a):

	for i in range(k):

		sig = np.asarray(sigmoid(x.dot(w.T)))
		R = calc_R(sig, x.shape[0])
		w_hat = w - np.linalg.inv(x.T.dot(R).dot(x) + (a + 1e-9)* np.identity(x.shape[1]) ).dot(((x.T.dot((sig - y).T)) + a * w))
		if check_stop(w_hat, w) < 1e-3:
			return w_hat, i
		w = w_hat

	return w, k

def pred_dist(w_map, x, a):

	sig = np.asarray(sigmoid(w_map.dot(x.T)))
	R = calc_R(sig, x.shape[0])

	Sn = np.linalg.inv(x.T.dot(R).dot(x) + a * np.identity(x.shape[1]) + 1e-9)

	sig_a = siga(x, Sn)
	mu = w_map.dot(x.T)

	pred = np.asarray(sigmoid(mu/np.sqrt(1 + (np.pi/8)*sig_a)))
	return pred

def siga(x, sn):

	sig_a = []
	for i in range(x.shape[0]):
		temp = x[i].dot(sn).dot(x[i])
		sig_a.append(temp)

	return np.asarray(sig_a)

def task_2(data):

	a_vals = []
	for d in range(0,len(data)):
		a = 0
		k = 1000
		x = np.c_[data[d][0], np.ones(data[d][0].shape[0])]
		train_size = int(x.shape[0] * .6)
		x = x[:train_size, :]
		y = data[d][1]
		y = y[:train_size]

		for l in range(10):
			w = np.zeros((x.shape[1]))
			w_map, k = Newton_raphson(x, y, w, k, a)
			gamma = calc_gamma(w_map, x, a)
			a = gamma / np.square(np.linalg.norm(w_map))
		a_vals.append([1, a])


	task_1(data, None, a_vals, "Task 2")
	return a_vals

def calc_gamma(w_map, x, a):

	sig = sigmoid(x.dot(w_map))
	R = calc_R(sig, x.shape[0])
	lam = np.linalg.eigh(x.T.dot(R).dot(x))[0]
	gamma = 0

	for i in range(len(lam)):
		gamma += lam[i]/(lam[i] + a)

	return gamma

def task_3(x, y, x_test, y_test):

	kernel = RBF()
	model = GaussianProcessClassifier(kernel=kernel).fit(x,y)
	err = 1 - model.score(x_test,y_test)
	return err



# DRIVER #
d, lrd = load_data()

print("task 1")
task_1(d,lrd, [[1],[1],[1],[1],[1]], "Task 1")

print("task 2")
a = task_2(d)

print("task 3")
task_1(d, lrd, a, "Task 3")