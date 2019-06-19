#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import glob
import pandas as pd


UTILS = "../utils/"
if UTILS not in sys.path:
    sys.path.append("../utils/")

from util import ql_ref_date
from util import recover_download_site_mab
from util import get_onlysite


def get_values(temp, dir_ctrl):
    dir_ctrl = os.path.join(temp, dir_ctrl)
    print("Reading",dir_ctrl)
    files = glob.glob(os.path.join(dir_ctrl, '*.log'))
    sarss = []
    sars = None
    file_id = 0
    for filename in sorted(files):
        ref = ql_ref_date(filename, split_text1='mab1-').replace('T', 'Z')
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
                        file_id += 1
                        sarss.append(sars)
                    sars = None
                elif '] rewards:' in _l:
                    sars['r'] = _l.split('] rewards:')[1].replace('\n', '').strip()
                    if 'None' in sars['r'] :
                        sars = None  # error, skip
                    else:
                        try:
                            ## find new state
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
    return sarss


def get_dataframe(TEMP='temp', MAB_DIR='MAB1'):
    # create a temporary dir
    if not os.path.exists(TEMP):
        os.mkdir(TEMP)
    # extract MAB results to TEMP
    files = glob.glob(os.path.join('..', MAB1_DIR, '/data/*.tar.xz'))
    for f in files:
        print("Extracting {}".format(os.path.basename(f)))
        s = "tar -C {} -xJf {}".format(TEMP, f)
        # print(s)
        os.system(s)
        
    sarss = get_values(TEMP, 'ctrl')
    print("Found", len(sarss))
    os.system("rm -fr {}".format(TEMP))
    
    data = pd.DataFrame(sarss)
    return data