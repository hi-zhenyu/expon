import time
import datetime

def get_current_time():
    current_time = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H:%M:%S')
    return current_time


if __name__ == "__main__":
    print(get_current_time())