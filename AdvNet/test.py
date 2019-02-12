import tensorflow as tf
tf.enable_eager_execution()
import loader
import config
import matplotlib.pyplot as plt
import numpy as np

conf=config.ConfigLoader("C:\\Users\\happy\\Desktop\\Venus\\AdvNet\\app\\abc\\config.txt")

load=loader.DataLoader("C:\\Users\\happy\\Desktop\\Venus\\AdvNet\\app\\abc\\structure_scene_box\\train",True)
data=load.load(config_loader=conf)


for in1, in2, out in data.take(1):
    x=[in1[0],in2[0],out[0]]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        # getting the pixel values between [0, 1] to plot it.
        drawable = np.array(x[i] * 0.5 + 0.5)
        drawable[drawable < 0] = 0
        drawable[drawable > 1] = 1

        # plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.imshow(drawable)
        plt.axis('off')
    plt.show()