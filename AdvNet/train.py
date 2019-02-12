# tensorflow deepling framework
import tensorflow as tf

tf.enable_eager_execution()

# third part dependency
import numpy as np
# normal dependency
import os
import time
import sys
import getopt

# other dependency
from model import cgan
import train_tool
import config
import loader

# the arguments are moved to the extra config file
# i just remain these code for fun
# BUFFER_SIZE = 230
# BATCH_SIZE = 1
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
# EPOCH=200
# SAVE_PERIOD=1
# DATA_PATH="C:/Users/happy/Desktop/AdvFuse/dataset/rio/rio_512/"
# CHECKPOINTS_DIR="./training_checkpoints/"
# LOG="log/"
# TRAINING_DEVICE="CPU"
# REMOVE_HISTORY_CHECKPOINTS=True
#
"""
@author yuwei
The load image function is removed to loader.py file, in which one or two input are both supported.
But more than 2 input are not supported, since currently, we don't think more than 2 input are necessary.
Maybe one day, 3 or more inputs will be used, and i will update the code anyway,  but not today.
"""
# finally delete these codes at @since 2019.1.24
"""
i don't know other situations, but to be honest, patch-gan is meanless in generating chinese charachers.
it is difficult to generate words and make me falling many time.
Do not trust anything totally, be criticized.
"""

"""
that's funny, i tried the patch gan before, but not very useful. today, i want to try again,
luckly, i didn't delete these codes, so i don't have to write it again.
maybe, it is still useful, i don;t know anyway
@since 2019.1.24. 22:30
"""


def patch(image, patch_size):
    """
    split the image to small patches
    :param image:
    :param patch_size:
    :return:
    """

    # get image size and calculate patch
    size = np.array(tf.shape(image))
    row_num = int(size[1] / patch_size)
    col_num = int(size[2] / patch_size)
    # patch list
    patch_list = []

    # split patch
    for i in range(row_num):
        for j in range(col_num):
            patch = image[:, patch_size * i:patch_size * (i + 1), patch_size * j:patch_size * (j + 1), :]
            patch_list.append(patch)

    return patch_list


"""
training iterator is from here, including global discriminator and local discriminator.
i don'y know whether it will be useful, just try.
@ since 2019.1.24
@ author yuwei
@ version 0.94
"""


def global_train_iterator(input_image, target, gen_tape, global_disc_tape,
                          generator, discriminator, generator_optimizer, discriminator_optimizer):
    """
    global train iterator, move to here, before @since 2019.1.24, it is in the train function body
    but the optimizer is become more and more and move to a specific function will be more clear
    @ author yuwei
    @ version 0.94 deep with global and local discriminator
    @ since 2019.1.24
    :param input_image:
    :param target:
    :param gen_tape:
    :param global_disc_tape:
    :param generator:
    :param discriminator:
    :param generator_optimizer:
    :param discriminator_optimizer:
    :return:
    """
    # print("start generate")
    gen_output = generator(input_image, training=True)

    # # print("start discriminator")
    disc_real_output = discriminator(input_image, target, training=True)

    # print("start generate output")
    disc_generated_output = discriminator(input_image, gen_output, training=True)

    # print("generator loss")
    gen_loss = cgan.generator_loss(disc_generated_output, gen_output, target)
    # print("disc loss")
    disc_loss = cgan.discriminator_loss(disc_real_output, disc_generated_output)

    # print("generator gradients ")
    generator_gradients = gen_tape.gradient(gen_loss, generator.variables)
    # print("disc gradients")
    discriminator_gradients = global_disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.variables))
    return gen_loss, disc_loss


"""
it is late now, and my laptop have no ability to run this code, with GPU MX940, thats not good
the server is running another code now, and the currounding update will try tomorrow
@since 2019.1.24. 23:01
@author yuwei
"""


def local_train_iterator(patch_size, input_image, target, gen_tape, local_disc_tape,
                         generator, discriminator, generator_optimizer, discriminator_optimizer):
    """
    :param patch_size:
    :param input_image:
    :param target:
    :param local_gen_tape:
    :param local_disc_tape:
    :param generator:
    :param discriminator:
    :param generator_optimizer:
    :param discriminator_optimizer:
    :return:
    """
    # generate output image
    gen_output = generator(input_image, training=True)

    # generate image to patch
    gen_output_patch = patch(gen_output, patch_size)
    # input image to patch
    input_patch = patch(input_image, patch_size)
    # real image to patch size
    target_patch = patch(target, patch_size)

    patch_num = len(gen_output_patch)

    # gen patch disc output list
    gen_disc_output_list = []
    # real path disc output list
    real_disc_output_list = []
    for i in range(patch_num):
        gen_disc_output_list.append(discriminator(input_patch[i], gen_output_patch[i], training=True))
        real_disc_output_list.append(discriminator(input_patch[i], target_patch[i], training=True))

    # gen total loss
    gen_total_loss = 0
    for i in range(patch_num):
        gen_total_loss = gen_total_loss + cgan.patch_generator_loss(gen_disc_output_list[i])

    # disc total loss
    disc_total_loss = 0
    for i in range(patch_num):
        disc_total_loss = disc_total_loss + cgan.patch_discrimator_loss(real_disc_output_list[i],
                                                                        gen_disc_output_list[i])

    # generator gradient
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.variables)
    discriminator_gradients = local_disc_tape.gradient(disc_total_loss, discriminator.variables)

    # optimizer
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.variables))

    return gen_total_loss, disc_total_loss


def train(train_dataset, test_dataset, config_loader):
    """
    train main loop
    :param train_dataset:
    :param test_dataset:
    :param config_loader:
    :return:
    """
    # start
    print("initial training progress....")

    # checkpoint instance
    print("initial checkpoint....")
    checkpoint_prefix = os.path.join(config_loader.checkpoints_dir, "ckpt")

    # The call function of Generator and Discriminator have been decorated
    # with tf.contrib.eager.defun()
    # We get a performance speedup if defun is used (~25 seconds per epoch)
    print("initial generator....")
    generator = cgan.Generator()
    print("initial global discriminator....")
    global_discriminator = cgan.Discriminator()

    # initial optimizer
    print("initial generator optimizer....")
    generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
    print("initial global discriminator optimizer....")
    global_discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

    # initial local discriminator and optimizer
    local_discriminator = None
    local_discriminator_optimizer = None
    # create patch mode
    if config_loader.patch_size >= 1:
        print("patch mode open, patch size={}".format(config_loader.patch_size))
        print("initial local discriminator....")
        local_discriminator = cgan.Discriminator()

        print("initial local discriminator optimizer....")
        local_discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         global_discriminator_optimizer=global_discriminator_optimizer,
                                         local_discriminator_optimizer=local_discriminator_optimizer,
                                         generator=generator,
                                         global_discriminator=global_discriminator,
                                         local_discriminator=local_discriminator)

    # only global mode
    else:
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                         global_discriminator_optimizer=global_discriminator_optimizer,
                                         generator=generator,
                                         global_discriminator=global_discriminator)

    print("initial train log....")
    log_tool = train_tool.LogTool(config_loader.log_dir, config_loader.save_period)

    # restoring the latest checkpoint in checkpoint_dir if necessary
    if config_loader.load_latest_checkpoint is True:
        checkpoint.restore(tf.train.latest_checkpoint(config_loader.checkpoints_dir))
        print("load latest checkpoint....")
        # print("{} epochs have been trained".format(tf.train.get_checkpoint_mtimes(checkpoint_prefixes=checkpoint_prefix)))

    # load patch size
    # patch_size=config_loader.patch_size
    # print("load patch size..., patch size={}".format(patch_size))

    # start training
    print("start training, epochs={}".format(config_loader.epoch))
    for epoch in range(1, config_loader.epoch):
        # initial time
        start = time.time()
        # initial input number
        image_num = 1
        # initial loss
        gen_loss = 0
        disc_loss = 0
        # each epoch, run all the images
        for input_image, target in train_dataset:
            print("input_image {}".format(image_num))
            # calculate input number
            image_num = image_num + 1

            with tf.GradientTape() as gen_tape, tf.GradientTape() as global_disc_tape:
                # global train
                gen_loss, disc_loss = global_train_iterator(input_image, target, gen_tape, global_disc_tape, generator,
                                                            global_discriminator, generator_optimizer,
                                                            global_discriminator_optimizer)

                # local train if necessary
            if config_loader.patch_size >= 1:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as local_disc_tape:
                    local_train_iterator(config_loader.patch_size, input_image, target, gen_tape, local_disc_tape,
                                         generator, local_discriminator, generator_optimizer,
                                         local_discriminator_optimizer)

        # save test result
        if epoch % config_loader.save_period == 0:
            # clear_output(wait=True)
            for input_image, ground_truth in test_dataset.take(1):
                log_tool.save_image_plt(model=generator, test_input=input_image, tar=ground_truth)
                log_tool.save_image(model=generator, test_input=input_image, tar=ground_truth)
                log_tool.save_loss(gen_loss=gen_loss, disc_loss=disc_loss)
                log_tool.update_epoch()

        # saving (checkpoint) the model every 20 epochs
        if epoch % config_loader.save_period == 0:
            # remove history checkpoints
            if config_loader.remove_history_checkpoints == True:
                train_tool.remove_history_checkpoints(config_loader.checkpoints_dir)
                print("remove history checkpoints...")

            # save the checkpoint
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("store current checkpoint successfully...")

            # #save generator model
            # generator.save_weights(os.path.join(config_loader.checkpoints_dir, "model.h5"))
            # print("store model successfully...")

        print('time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))


def main(config_loader):
    # load the train dataset
    train_dir = os.path.join(config_loader.data_dir, "train")
    train_loader = loader.DataLoader(train_dir, is_training=True)
    train_dataset = train_loader.load(config_loader)
    # load the test dataset
    test_dir = os.path.join(config_loader.data_dir, "test")
    test_loader = loader.DataLoader(test_dir, is_training=False)
    test_dataset = test_loader.load(config_loader)

    # train
    if config_loader.training_device in ("CPU"):
        with tf.device("/cpu:0"):
            train(train_dataset=train_dataset, test_dataset=test_dataset, config_loader=config_loader)
    else:
        with tf.device("/gpu:0"):
            train(train_dataset=train_dataset, test_dataset=test_dataset, config_loader=config_loader)


# bootstrap the train progress
if __name__ == "__main__":

    # default runtime environment
    env = "./"
    # default config file
    config_file = "config.txt"

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "he:c:", ["help", "env=", "config="])
        print(options)
        for key, value in options:
            if key in ("-c", "--config"):
                config_file = value
            if key in ("-e", "--env"):
                env = value
            if key in ("-h", "--help"):
                print("-h -e dic -c config.txt")

    # set current runtime environment
    os.chdir(env)
    print(os.getcwd())
    # load config file
    config_loader = config.ConfigLoader(config_file)
    # start training
    main(config_loader=config_loader)
