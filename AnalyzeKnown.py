#!/usr/bin/python3

import sys
import os
import time
import argparse
import json
import scipy.stats
import scipy
import math
import copy
import signal
import struct
import shutil
import subprocess
import shlex
import zlib
import inspect
import re
import pprint
import pickle
import locale

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
from collections import OrderedDict
from collections import Counter
from natsort import natsorted


KNOWN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_mp_list.txt")

def addArgs():
    p = argparse.ArgumentParser()
    p.add_argument("--histo_bins", required=False, type=int, default=10,
                   help="Number of bins in histo")
    p.add_argument("--last_N", required=False, type=int, default=2,
                   help="Last N to sum")
    p.add_argument("--no_plot", action="store_true",
                   help="Don't plot it")
    return p.parse_args()

def writeFlush(str_, stderr_=False):
    if not stderr_:
        sys.stdout.write(str_ + "\n")
        sys.stdout.flush()
    else:
        sys.stderr.write(str_ + "\n")
        sys.stderr.flush()

def runShellCommand(cmd_):
    s_output = ""
    try:
        pid = subprocess.Popen(cmd_, stdout=subprocess.PIPE, shell=True)
        s_output, s_err = pid.communicate()
        if not s_output and not s_err:
            return ""
    except:
        writeFlush("Exception during call: " + cmd_, True)
        return ""
    finally:
        pass
    if s_err:
        writeFlush("Error during call: " + cmd_ + ": " + s_err, True)
    s_out = "" 
    t = type(s_output) if s_output else int
    if t != str:
        if str(t).find("bytes") > -1:
             s_out = s_output.decode( "unicode-escape" ).replace("\r\n", "\n")
    elif str(t).find("str"):
        s_out = s_output.decode( "ascii" ).replace("\r\n", "\n")
    return s_out

def fiboNext(l_seq_):
    return sum(l_seq_[-2:])

def sumLastN(l_seq_, N_):
    r = sum(l_seq_[-N_:]) if N_ < len(l_seq_) else sum(l_seq_)
    return r

def analyzeKnown(args_):
    l_ratio = list()
    l_sqrt_sum = list()
    l_last_N_ratio = list()
    with open(KNOWN_FILE, 'r') as f:
        l_kp = f.read().split('\n')
        f.close()
        l_kp = [float(a) for a in l_kp if len(a)]
        for i, kp in enumerate(l_kp):
            if i < args_.last_N:
                pass
            if not i:
                continue
            # print("i:{0} | kp:{1} | sum:{2}".format(i, kp, sum(l_kp[:i])))
            # print("i:{0} | kp:{1} | fibo:{2}".format(i, kp, l_kp[i-2:i]))
            l_ratio.append(kp / sum(l_kp[:i]))
            l_sqrt_sum.append(math.sqrt(kp / sum(l_kp[:i])))
            l_last_N_ratio.append(kp / sumLastN(l_kp[:i], args_.last_N))
    mpl.rcParams['toolbar'] = 'None'
    plt.style.use('dark_background')
    fig, ax = plt.subplots(3,1, squeeze=True, figsize=(16,11), sharex='none', facecolor='Black')
    ax[0].plot(l_ratio)
    ax[1].plot(l_last_N_ratio)
    ax[0].set_ylabel("Sum Ratio", color="orange")
    ax[1].set_ylabel("Last N Ratio", color="orange")
    ax[1].set_xlabel("Known Prime Index", color="orange")
    plt.draw()
    n, bins, patches = ax[2].hist(l_last_N_ratio, args_.histo_bins, color='green')
    ax[2].set_ylabel("Ratio Last N Histo", color="orange")
    l_cum_sum = np.cumsum(np.array(n))
    sum_last_N_now = sumLastN(l_kp, args_.last_N)
    # print(l_kp)
    # print("last kp:{0} | sum last N now:{1} | min last N ratio:{2}".format(l_kp[-1], sum_last_N_now, min(l_last_N_ratio)))
    lower_bound_next = int(sum_last_N_now * min(l_last_N_ratio) + 0.5)
    lower_bound_next = int(l_kp[-1] + 1) if lower_bound_next <= l_kp[-1] else lower_bound_next
    likely_90_perc_index = 0
    sum_n = sum(n)
    # print(l_cum_sum / sum_n)
    while l_cum_sum[likely_90_perc_index] / sum_n < 0.9:
        # print(l_cum_sum[likely_90_perc_index] / sum_n)
        likely_90_perc_index += 1
    likely_90_perc_index += 1
    # print(l_cum_sum[likely_90_perc_index] / sum_n)
    likely_upper_bound_next = int(sum_last_N_now * bins[likely_90_perc_index] + 0.5)
    writeFlush("likely lower bound: {0} | 90%% likely upper_bound: {1}".format(lower_bound_next, likely_upper_bound_next))
    if args_.no_plot:
        sys.exit(1)
    ax2_2 = ax[2].twinx()
    delta_bin = (bins[1] - bins[0]) / 2
    ax2_2.plot(bins[:-1] + delta_bin, l_cum_sum / sum(n), 'b')
    l_lims = ax2_2.axis()
    ax2_2.set_ylim(0, l_lims[3])
    ax[2].set_xlabel("Last N Ratio: N = " + str(args_.last_N), color="orange")
    plt.suptitle("Ratio of Known Mersenne Primes vs Variety of Metrics", color="skyblue")
    plt.show()

def signalHandler(signal_, frame):
    writeFlush("\nkilled by ctrl-C\n")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, signalHandler)
    args = addArgs()

    analyzeKnown(args)

if __name__=="__main__":
    sys.exit(main())
