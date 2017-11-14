import os
import sys
from datetime import datetime


class console:

    def __init__(self):
        pass

    # print message that can be updated in the console
    def pm_flush(self, message):
        t = datetime.now()
        sys.stdout.write('\r[!] ({})      {}'.format(t, message))
        sys.stdout.flush()
        return None

    def pm_stat(self, message):
        t = datetime.now()
        print('\r[!] ({})      {}'.format(t, message))
        return None

    def pm_err(self, message):
        t = datetime.now()
        print('[X] ({})      {}'.format(t, message))
        return None