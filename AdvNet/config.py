# key
INPUT_INTRODUCTION = "#The following are input arguments"
BUFFER_SIZE = "buffer_size"
BATCH_SIZE = "batch_size"
IMAGE_WIDTH = "image_width"
IMAGE_HEIGHT = "image_height"
INPUT_NUM = "input_number"
DATA_FLIP = "data_flip"
TRAIN_INTRODUCTION = "#The followong are training arguments"
EPOCH = "epoch"
PATCH_SIZE = "patch_size"
SAVE_PERIOD = "save_period"
DATA_DIR = "data_dir"
CHECKPOINTS_DIR = "checkpoints_dir"
LOG_DIR = "log_dir"
TRAINING_DEVICE = "training_device"
REMOVE_HISTORY_CHECKPOINTS = "remove_history_checkpoints"
LOAD_LATEST_CHECKPOINTS = "load_latest_checkpoints"
SPLIT = "="


def create_config_temple(config_dir, buffer_size=0, batch_size=1, image_width=256, image_height=256, input_num=1,
                         data_flip=0, patch_size=0,
                         epoch=2000, save_period=5, data_dir="./data/", checkpoints_dir="./training_checkpoints/",
                         log_dir="./log/", training_device="GPU", remove_history_checkpoints=True,
                         load_latest_checkpoints=True):
    config_lines = [BUFFER_SIZE, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, INPUT_NUM, DATA_FLIP, EPOCH, PATCH_SIZE,
                    SAVE_PERIOD, DATA_DIR, CHECKPOINTS_DIR, LOG_DIR,
                    TRAINING_DEVICE, REMOVE_HISTORY_CHECKPOINTS, LOAD_LATEST_CHECKPOINTS]

    value = [buffer_size, batch_size, image_width, image_height, input_num, data_flip, epoch, patch_size, save_period,
             data_dir,
             checkpoints_dir,
             log_dir, training_device, remove_history_checkpoints, load_latest_checkpoints]

    for i in range(len(config_lines)):
        config_lines[i] = "{} {} {}\n".format(config_lines[i], SPLIT, value[i])

    with open(config_dir, "w") as f:
        f.writelines(config_lines)


class ConfigLoader:

    def __init__(self, config_dir):
        """
        config dir
        :param config_dir:
        """
        # read file
        with open(config_dir, "r") as f:
            lines = f.readlines()

        # load
        self.config_list = dict()
        for line in lines:
            # "#"is used to make description
            # @since 2019.1.18
            # @version 0.93
            # @author yuwei
            if "#" not in line:
                value = line.strip().split(SPLIT)
                self.config_list[value[0].strip()] = value[1].strip()

        # parse
        self.buffer_size = int(self.config_list[BUFFER_SIZE])
        self.batch_size = int(self.config_list[BATCH_SIZE])
        self.image_width = int(self.config_list[IMAGE_WIDTH])
        self.image_height = int(self.config_list[IMAGE_HEIGHT])
        self.input_num = int(self.config_list[INPUT_NUM])
        self.data_flip = float(self.config_list[DATA_FLIP])
        self.epoch = int(self.config_list[EPOCH])
        self.patch_size=int(self.config_list[PATCH_SIZE])
        self.save_period = int(self.config_list[SAVE_PERIOD])
        self.data_dir = self.config_list[DATA_DIR]
        self.checkpoints_dir = self.config_list[CHECKPOINTS_DIR]
        self.log_dir = self.config_list[LOG_DIR]
        self.training_device = self.config_list[TRAINING_DEVICE]

        # if self.config_list[REMOVE_HISTORY_CHECKPOINTS] == "True" or self.config_list[
        #     REMOVE_HISTORY_CHECKPOINTS] == "true":
        if self.config_list[REMOVE_HISTORY_CHECKPOINTS] in ("True", "true"):
            self.remove_history_checkpoints = True
        else:
            self.remove_history_checkpoints = False

        if self.config_list[LOAD_LATEST_CHECKPOINTS] in ("True", "true"):
            self.load_latest_checkpoint = True
        else:
            self.load_latest_checkpoint = False

        # # input concat
        # if self.config_list[INPUT_CONCAT] in ("True", "true"):
        #     self.input_concat = True
        # else:
        #     self.input_concat = False
