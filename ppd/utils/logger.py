import os
class Log:
    log_on = True 
    used_tags = dict()  # To keep track of used tags
    _is_main_cached = None  # Cache to store the main process check result

    @staticmethod
    def is_main_process():
        if Log._is_main_cached is not None:
            return Log._is_main_cached
        try:
            from pytorch_lightning.utilities import rank_zero_only
            if rank_zero_only.rank == 0:
                Log._is_main_cached = True
            else:
                Log._is_main_cached = False
        except:
            Log._is_main_cached = True
        return Log._is_main_cached

    @staticmethod
    def _should_log(tag):
        if not Log.log_on:
            return False
        if not Log.is_main_process():
            return False
        if tag is None:
            return True
        if '__' in tag:
            num = int(tag.split('__')[-1])
            tag = tag.split('__')[0]  
        else:
            num = 3 

        if tag not in Log.used_tags:
            Log.used_tags[tag] = num
        Log.used_tags[tag] -= 1
        if Log.used_tags[tag] >= 0:
            return True
        else:
            return False

    @staticmethod
    def info(*args, tag=None):
        if Log._should_log(tag):
            print("\033[1;32m[INFO]\033[0;0m", *args)

    @staticmethod
    def warn(*args, tag=None):
        if Log._should_log(tag):
            print("\033[1;35m[WARN]\033[0;0m", *args)

    @staticmethod
    def error(*args, tag=None):
        print("\033[1;31m[ERROR]\033[0;0m", *args)

    @staticmethod
    def debug(*args, tag=None):
        if Log._should_log(tag) and 'HT_DEBUG' in os.environ and os.environ['HT_DEBUG'] == '1':
            print("\033[1;33m[DEBUG]\033[0;0m", *args)



def monitor_process_wrapper(func):
    """The wrapper will print a log both before and after the wrapped function runned."""
    def wrapped(*args, **kwargs):
        Log.info(f'"{func.__name__}()" begin...')
        ret_value = func(*args, **kwargs)
        Log.info(f'"{func.__name__}()" end...')
        return ret_value
    return wrapped

def monitor_class_process_wrapper(func):
    """The wrapper will print a log both before and after the wrapped function runned."""
    def wrapped(self, *args, **kwargs):
        Log.info(f'"{self.__class__.__name__}.{func.__name__}()" begin...')
        ret_value = func(self, *args, **kwargs)
        Log.info(f'"{self.__class__.__name__}.{func.__name__}()" end...')
        return ret_value
    return wrapped
