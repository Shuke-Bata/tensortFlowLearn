# -*- coding: utf-8 -*-
"""
CNN卷积神经网络实现手写数字识别

"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

class CNN(object):
  def __init__(self):
    # 调用Sequential模型
    model = models.Sequential()

    # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小,activation指的是激活函数
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # 第2层卷积，卷积核大小为3*3，64个
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # 第3层卷积，卷积核大小为3*3，64个
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # 用来打印我们定义的模型的结构。
    model.summary()

    self.model = model

class DataSource(object):
  def __init__(self):
    # mnist数据集自动下载
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # 6万张训练图片，1万张测试图片
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    # 像素值映射到 0 - 1 之间
    train_images, test_images = train_images / 255.0, test_images / 255.0

    self.train_images, self.train_labels = train_images, train_labels
    self.test_images, self.test_labels = test_images, test_labels

class Train:
  def __init__(self):
    self.cnn = CNN()
    self.data = DataSource()

  # 展示训练历史
  def show_train_history(self, train_history, train, validation):
    plt.plot(train_history.history[train]) #绘制训练数据的执行结果
    plt.plot(train_history.history[validation]) #绘制验证数据的执行结果
    plt.title('Train History') #图标题 
    plt.xlabel('epoch') #x轴标签
    plt.ylabel(train) #y轴标签
    plt.legend(['train','validation'],loc='upper left') #添加左上角图例
  
  def train(self):
    check_path = '/content/ckpt/cp-{epoch:04d}.ckpt'

    # period 每隔5 epoch保存一次
    save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

    # 优化器采用的是adam优化器，loss这里采用的交叉熵损失，评估方式为准确率
    self.cnn.model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
    # epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
    # validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。
    # train_history = self.cnn.model.fit(self.data.train_images, self.data.train_labels, validation_split=0.1, epochs=10, callbacks=[save_model_cb])
    train_history = self.cnn.model.fit(self.data.train_images, self.data.train_labels, validation_split=0.2, epochs=5, callbacks=[save_model_cb])

    self.show_train_history(train_history,'accuracy','val_accuracy')

    test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
    print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
  app = Train()
  app.train()

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

!apt from train import CNN

class Predict(object):
  def __init__(self):
    latest = tf.train.latest_checkpoint('/content/ckpt/')
    self.cnn = CNN()
    # 恢复网络权重
    self.cnn.model.load_weights(latest)

  def predict(self, image_path, expected_digtil):
    # 以黑白方式读取图片
    img = Image.open(image_path).convert('L')
    flatten_img = np.reshape(img, (28, 28, 1))
    x = np.array([1 - flatten_img])

    # API refer: https://keras.io/models/model/
    y = self.cnn.model.predict(x)

    # 因为x只传入了一张图片，取y[0]即可
    plt.imshow(img)

    # np.argmax()取得最大值的下标，即代表的数字
    predict_digtil = np.argmax(y[0])
    title = 'real digtil==>%d predict_digtil==>%d' %(expected_digtil, predict_digtil)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
  app = Predict()
  app.predict('/content/0.png', 0)
  app.predict('/content/1.png', 1)
  app.predict('/content/2.png', 2)
  app.predict('/content/3.png', 3)
  app.predict('/content/4.png', 4)
  app.predict('/content/5.png', 5)
  app.predict('/content/6.png', 6)
  app.predict('/content/7.png', 7)
  app.predict('/content/8.png', 8)
  app.predict('/content/9.png', 9)