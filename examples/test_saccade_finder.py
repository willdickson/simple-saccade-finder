import h5py
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from simple_saccade_finder import find_saccades

delta_wba_to_volts = 4.71/90.0

param = { 
         'hp_fcut'    : 2.0,
         'lp_fcut'    : 8.0,
         't_start'    : 0.0, 
         'threshold'  : 0.3,
         'hysteresis' : 0.6,
         'duration'   : 0.01,
         'refractory' : 1.0, 
         }

data_file = pathlib.Path('saccade_data', 'ten_kHz_data_251121_18.hdf5')
data = h5py.File(data_file, 'r')
t = np.array(data['ThorSync']['time_10k'])
v = np.array(data['ThorSync']['dWA_raw_10k'])*delta_wba_to_volts

saccade_info = find_saccades(t, v, param)

if 1:
    hp_fcut = param['hp_fcut']
    lp_fcut = param['lp_fcut']

    x_bp = saccade_info['x_bp']
    saccade_tvals = saccade_info['saccade_tvals']
    saccade_ivals = saccade_info['saccade_ivals']
    saccade_xvals = saccade_info['saccade_xvals']

    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(t, v, 'b')
    ax[0].grid(True)
    ax[0].set_xlabel('t (sec)')
    ax[0].set_ylabel(r'$\Delta$WBA')
    ax[0].set_title(f'hp {hp_fcut:0.1f} (Hz), lp {lp_fcut:0.1f} (Hz)')
    
    ax[1].plot(t, x_bp, 'g')
    ax[1].plot(saccade_tvals, saccade_xvals, '.r')
    ax[1].grid(True)
    ax[1].set_xlabel('t (sec)')
    ax[1].set_ylabel(r'$\Delta$WBA filt')

    plt.show()
    




