import errno
import os

import shutil


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def execute(command):
    os.system(command)


def rmdir(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            pass
        else:
            raise


def copy_dir(source, dest):
    rmdir(dest)
    shutil.copytree(source, dest)


def copy_file(source, dest):
    shutil.copyfile(source, dest)


def current_dir():
    return os.getcwd()


def change_dir(path):
    os.chdir(path)
