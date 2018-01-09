#!/usr/bin/env python2

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import paratemp.energy_histo as eh
from paratemp import cd
from glob import glob
from shutil import copyfile, rmtree


if sys.version_info.major != 2:
    raise ImportError('Last I checked, paratemp.energyHisto which uses '
                      'MDAnalysis will not work with python 3.*.')


plt.style.use('seaborn-colorblind')

regex_time = re.compile(r'Replica exchange at step \d+ time (\d+\.\d+)')

start_dir = os.getcwd()
temp_dir = "cctempdir"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

edrs = glob('*.edr')
log_file = glob('*out*.log')[0]
colvar_files = glob('COLVAR*')

files_to_copy = edrs + [log_file] + colvar_files

for filename in files_to_copy:
    copyfile(filename, temp_dir+'/'+filename)

with cd(temp_dir):
    if len(colvar_files) == 0:
        print('No COLVAR files; ignoring that part.')
        with open(log_file, 'r') as f_log:
            final_time = None
            for line in f_log:
                try:
                    final_time = regex_time.match(line).group(1)
                    final_time = '{:.0f}'.format(float(final_time)/1000)
                except AttributeError:
                    continue
            if final_time is None:
                print("Couldn't find final time from log file.")
                final_time = 'unk'
    else:
        colvars = dict()
        for i in range(16):
            fname = 'COLVAR.' + str(i)
            df = pd.read_table(fname,
                               sep='\s+',
                               header=0,
                               names=('time', 'dm1', 'dm2'),
                               skiprows=1,
                               index_col=0)
            colvars[i] = df
        final_time = str(int(colvars[0].index[-1]/1000))
        fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(5, 5))
        for i in range(16):
            ax = axes.flat[i]
            ax.hist2d(colvars[i]['dm1'], colvars[i]['dm2'])
            ax.set_aspect('equal', 'box-forced')
        cv_hists = fig
        cv_hists.savefig('converg-check-'+final_time+'ns-2d-hists.pdf')
    eh.make_basic_plots(save_base_name='converg-check-'+final_time+'ns',
                        display=False,
                        logfile=log_file)
    pdfs = glob('*.pdf')
    for pdf in pdfs:
        copyfile(pdf, start_dir+'/'+pdf)
    print('Generated and copied files '+str(pdfs)+' to '+start_dir)

rmtree(temp_dir)


