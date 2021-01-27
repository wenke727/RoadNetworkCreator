import os


os.popen('ps -ef | grep LSTR').read().split('\n')