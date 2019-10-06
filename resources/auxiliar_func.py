import numpy as np
import time


def time_to_num(time_str, index='hms'):
    hh, mm, ss = map(int, time_str.split(':'))
    if index == 'hms':
        return ss + 60 * (mm + 60 * hh)
    elif index == 'hm':
        return mm + 60 * (60 * hh)


def num_to_time(time_int):
    return time.strftime('%H:%M:%S', time.gmtime(time_int))


def shift(arr, n):
    if n == 0: return arr
    arr_len = len(arr)
    return [arr[(i + n) % arr_len] for i in range(arr_len)]


def calc_ma(data, period):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + period - 1:]
    empty_list = [None] * (j + period - 1)
    sub_result = [np.mean(data[i - period + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)

