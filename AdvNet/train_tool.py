import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def parse_last_epoch(file):
    """
    parse last epoch, invoked by init function
    :param file:
    :return:
    """
    if os.path.exists(file) is False:
        return 0
    else:
        with open(file, "r") as f:
            lines = f.readlines()
            last_line = lines[-1]
            return int(last_line.split(",")[0].split("=")[1])



"""
actually, log epoch must be vary instead of 1
@fixed at version 0.93
@author yuwei
@since 2019.1.17

"""


class LogTool:

    def __init__(self, log_dir, save_period):
        """
        initial directionary
        :param log_dir:
        :param save_period
        """
        self.log_dir = log_dir
        self.save_period = save_period;
        self.image_dir = os.path.join(log_dir, "result_image")
        self.train_log_dir = os.path.join(log_dir, "train_log.txt")
        # create dir if not exist
        if os.path.exists(self.log_dir) is False:
            os.mkdir(log_dir)
        if os.path.exists(self.image_dir) is False:
            os.mkdir(self.image_dir)
        # upate current epoch
        self.current_epoch = parse_last_epoch(self.train_log_dir) + self.save_period

    def save_image_plt(self, model, test_input, tar):
        # the training=True is intentional here since
        # we want the batch statistics while running the model
        # on the test dataset. If we use training=False, we will get
        # the accumulated statistics learned from the training dataset
        # (which we don't want)
        prediction = model(test_input, training=True)

        # tf shape is a tensor, and take out the element, it is essential to change to numpy array and compare the diminsion.
        # fixed in version 0.93
        # @author yuwei
        # @since 2019.1.17
        if np.array(tf.shape(test_input))[3] == 3:
            display_list = [test_input[0], tar[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']
        else:
            display_list = [test_input[0, :, :, 0:3], test_input[0, :, :, 3:6], tar[0], prediction[0]]
            title = ['Input Image 1', "Input Image 2", 'Ground Truth', 'Predicted Image']

        grid_num = len(title)
        for i in range(grid_num):
            plt.subplot(1, grid_num, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            drawable = np.array(display_list[i] * 0.5 + 0.5)
            drawable[drawable < 0] = 0
            drawable[drawable > 1] = 1

            # plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.imshow(drawable)
            plt.axis('off')
        output_image_file = os.path.join(self.image_dir, str(self.current_epoch) + ".png")
        plt.savefig(output_image_file)
        plt.close()

    def save_image(self, model, test_input, tar):
        """
        save the images independently, instead of saving in an figure by plt
        these images will be shown by log server on other remote devices, such as a browser
        :param model:
        :param test_input:
        :param tar:
        :return:
        """
        # get image from -0.5,0.5
        prediction = model(test_input, training=True)
        # the title indicate the input image, ground truth and prediction

        # this is the same situation in save_image_plt
        if np.array(tf.shape(test_input))[3] == 3:
            display_list = [test_input[0], tar[0], prediction[0]]
            title = ['IN1', 'GT', 'PR']
        else:
            display_list = [test_input[0, :, :, 0:3], test_input[0, :, :, 3:6], tar[0], prediction[0]]
            title = ['IN1', "IN2", 'GT', 'PR']

        for i in range(len(title)):
            # switch the list to [0,1]
            drawable = np.array(display_list[i] * 0.5 + 0.5)
            drawable[drawable < 0] = 0
            drawable[drawable > 1] = 1
            # save dir
            dir = os.path.join(self.image_dir, str(self.current_epoch) + "_" + title[i] + ".png")
            plt.imsave(dir, drawable)
            plt.close()

    def save_loss(self, gen_loss, disc_loss):
        """
        save loss
        :param gen_loss:
        :param disc_loss:
        :return:
        """
        with open(self.train_log_dir, "a") as f:
            line = "epoch={},timestamp={},gen_loss={},disc_loss={}\n".format(self.current_epoch, time.strftime("%b-%d-%Y-%H:%M:%S", time.localtime()), gen_loss,
                                                                             disc_loss)
            f.writelines(line)

    def update_epoch(self):
        """
        update epoch
        :return:
        """
        # next epcoh
        self.current_epoch = self.current_epoch + self.save_period


def parse_log(log_dir):
    """
    parse log, current includes generator loss and disc loss
    :param log_dir:
    :return:
    """
    with open(log_dir, "r") as f:
        lines = f.readlines()

    gen_loss = []
    disc_loss = []
    for line in lines:
        line = line.strip().split(",")
        gen_loss.append(float(line[2].split("=")[1]))
        disc_loss.append(float(line[3].split("=")[1]))

    return gen_loss, disc_loss


def paint_loss(log_dir):
    """
    print loss
    :param log_dir:
    :return:
    """
    gen_loss, disc_loss = parse_log(log_dir)
    plt.plot(gen_loss, "b")
    plt.title("generator loss")
    # plt.show()

    plt.figure(2)
    plt.plot(disc_loss, "b")
    plt.title("discrimator loss")
    plt.show()


def remove_history_checkpoints(dir):
    """
    remove history checkpoints and remain nothing
    :param dir:
    :return:
    """
    if os.path.exists(dir) is False:
        return
    # get all checkpoints
    file_list = []
    for file in os.listdir(dir):
        file_list.append(file)

    # @since 2018.12.29 this method will remove all the checkpoints
    # sort
    # file_list.sort(reverse=True)

    # if only a check point is in the directionary, do not delete it
    # if len(file_list)<=3:
    #   return

    for each in file_list:
        # do not remove checkpoint file
        # if file_list[i] == "checkpoint":
        #    continue

        # remove history
        remove_file = os.path.join(dir, each)
        os.remove(path=remove_file)
