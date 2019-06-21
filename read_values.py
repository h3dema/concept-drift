#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import glob
import pandas as pd
import sys
import re


UTILS = "../utils/"
if UTILS not in sys.path:
    sys.path.append("../utils/")


from util import ql_ref_date
from util import recover_download_site_mab
from util import get_onlysite


def get_values(temp, dir_ctrl, type=1):
    dir_ctrl = os.path.join(temp, dir_ctrl)
    print("Reading", dir_ctrl)
    files = glob.glob(os.path.join(dir_ctrl, '*.log'))
    print("Going to process {} files".format(len(files)))
    sarss = []
    sars = None
    file_id = 0
    for filename in sorted(files):
        if type == 1:
            ref = ql_ref_date(filename, split_text1='mab1-').replace('T', 'Z')
        elif type == 2:
            ref = re.match('.*mab2-(.*).log', filename, re.I).groups()
            if len(ref) == 0:
                # no match
                continue
            ref = ref[0]
        elif type == 3:
            ref = ql_ref_date(filename, split_text1='mab3-').replace('T', 'Z')
        else:
            continue

        site1 = get_onlysite(recover_download_site_mab(relative_path=temp, sta=1, ref_date=ref, prefix='sta-mab-'))
        site2 = get_onlysite(recover_download_site_mab(relative_path=temp, sta=2, ref_date=ref, prefix='sta-mab-'))

        with open(filename, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            _l = lines[i]
            if 'iteration' in _l:
                sars = None
            if 'Frequency: ' in _l:
                f = re.findall(r"[-+]?\d*\.\d+|\d+", _l)
                if len(f) == 10:
                    # ok
                    sars = dict(zip(['Frequency', 'Medium busy', 'Busy time', 'Active time'], f[6:]))
            elif sars is not None:
                # ok, got at least Frequency...
                if '] AP0 txpower:' in _l:
                    f = re.findall(r"[-+]?\d*\.\d+|\d+", _l)
                    if len(f) == 11:
                        sars.update(dict(zip(['txpower', 'new_txpower', 'channel', 'new_channel'], f[7:])))
                    if 'new Medium busy' in sars:
                        sars.update({'file_id': file_id})
                        sars.update({'sites': (site1, site2)})
                        sarss.append(sars)
                    sars = None
                elif '] rewards:' in _l:
                    sars['r'] = _l.split('] rewards:')[1].replace('\n', '').strip()
                    if 'None' in sars['r']:
                        sars = None  # error, skip
                    else:
                        try:
                            # find new state
                            for j in range(1, 10):
                                _ll = lines[i + j]
                                if 'Frequency: ' in _ll:
                                    f = re.findall(r"[-+]?\d*\.\d+|\d+", _ll)
                                    if len(f) == 10:
                                        # ok
                                        new_sars = dict(zip(['new Medium busy', 'new Busy time', 'new Active time'], f[7:]))
                                        sars.update(new_sars)
                                    break
                        except (KeyError, IndexError):
                            sars = None  # error skip
        file_id += 1
    return sarss


def get_dataframe(TEMP='temp', __directory='MAB1', extract_files=True):
    if extract_files:
        # create a temporary dir
        if not os.path.exists(TEMP):
            os.mkdir(TEMP)
        # extract MAB results to TEMP
        files = glob.glob('../{}/data/*.tar.xz'.format(__directory))
        for f in files:
            print("Extracting {}".format(os.path.basename(f)))
            s = "tar -C {} -xJf {}".format(TEMP, f)
            # print(s)
            os.system(s)

    mab_type = {'MAB1': 1, 'MAB2.1': 2, 'MAB3.1': 3}
    sarss = get_values(TEMP, 'ctrl', mab_type[__directory])

    print("Found", len(sarss))
    if extract_files:
        os.system("rm -fr {}".format(TEMP))

    data = pd.DataFrame(sarss)
    data['r'] = data['r'].astype('float')
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read MAB files to collect data with sars format.')
    parser.add_argument('--output', type=str, default='sarss.h5', help='file to output sars')
    parser.add_argument('--dir', type=str, default='MAB1', help='MAB dir where the tar.xz file are')
    parser.add_argument('--temp', type=str, default='temp', help='temp dir')

    args = parser.parse_args()
    data = get_dataframe(TEMP=args.temp, __directory=args.dir)
    print("Read {}".format(data.shape))
    print("Saving data to {}".format(args.output))
    data.to_hdf(args.output, key='sarss', mode='w')
