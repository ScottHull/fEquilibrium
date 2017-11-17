import os
import sys
from datetime import datetime


class console:

    def __init__(clf):
        pass

    # print message that can be updated in the console
    @classmethod
    def pm_flush(clf, message):
        t = datetime.now()
        sys.stdout.write('\r[!] ({})      {}'.format(t, message))
        sys.stdout.flush()
        return None

    @classmethod
    def pm_stat(clf, message):
        t = datetime.now()
        print('\r[!] ({})      {}'.format(t, message))
        return None

    @classmethod
    def pm_err(clf, message):
        t = datetime.now()
        print('[X] ({})      {}'.format(t, message))
        return None

    @classmethod
    def pm_header(clf, message):
        print('{}'.format(message))
        return None