import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

train_loss_cnn, train_accuracy_cnn = pickle.load(open('result/cnn_train.p', 'rb'))
train_loss_capsule, train_accuracy_capsule = pickle.load(open('result/capsule_train.p', 'rb'))
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(range(10), train_loss_cnn, label='cnn train loss')
axarr[0].plot(range(10), train_loss_capsule, label='capsule train loss')
axarr[0].set_title('Training loss and accuracy')
axarr[0].set_xlabel('Epoch')
axarr[0].set_ylabel('train loss')
axarr[0].legend()
axarr[1].plot(range(10), train_accuracy_cnn, label='cnn train accuracy')
axarr[1].plot(range(10), train_accuracy_capsule, label='capsule train accuracy')
axarr[0].set_ylabel('train accuracy')
axarr[1].legend()
plt.savefig("img/train_curve.png")
