import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

train_loss_cnn, train_accuracy_cnn = pickle.load(
	open(config.CNN_TRAIN_RESULT_PATH, 'rb'))
train_loss_capsule, train_accuracy_capsule = pickle.load(
	open(config.CAPSULE_TRAIN_RESULT_PATH, 'rb'))

f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(range(10), train_loss_cnn, label='cnn train loss')
axarr[1].plot(range(10), train_loss_capsule, label='capsule train loss')
axarr[0].set_title('Training loss and accuracy')
axarr[0].set_xlabel('Epoch')
axarr[0].set_ylabel('train loss')
axarr[1].set_ylabel('train loss')
axarr[0].legend()
axarr[1].legend()
axarr[2].plot(range(10), train_accuracy_cnn, label='cnn train accuracy')
axarr[2].plot(range(10), train_accuracy_capsule, label='capsule train accuracy')
axarr[2].set_ylabel('train accuracy')
axarr[2].legend()
plt.savefig("img/train_curve.png")
