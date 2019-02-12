import os
import sys
import getopt
import config


def create_train_bootstrap():
    """
    create train bootstrap
    :return:
    """
    cmd = ["python ../../train.py --env=./ --config=config.txt\r\n", "@pause\r\n"]
    with open("train.bat", "w") as f:
        f.writelines(cmd)


def create_app(app_name):
    """
    create an empty app
    :param app_name:
    :return:
    """
    # create dictionary
    app_dir = os.path.join("./app/", app_name)

    # check is existed
    if os.path.exists(app_dir):
        return "error app name, the app = {} is already existed".format(app_name)

    os.mkdir(app_dir)

    # switch to app dir
    os.chdir(app_dir)
    # create config file temple
    config.create_config_temple("config.txt")
    # create bootstrap train.bat
    create_train_bootstrap()

    return "create app successfully, name={}".format(app_name)


if __name__ == "__main__":
    # parse input arguments if exist

    app_name = None

    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "hn:", ["help", "name="])
        print(options)
        for key, value in options:
            if key in ("-n", "--name"):
                app_name = value

    if app_name is not None:
        msg = create_app(app_name)
        print(msg)
    else:
        print("no name is selected, please enter you name by --name= or -n ")
