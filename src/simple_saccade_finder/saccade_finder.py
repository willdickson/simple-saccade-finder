import numpy as np
from .filter import Bandpass
from enum import Enum

class FinderState(Enum):
    """ Operating states for the SaccadeFinder class """
    READY = 0
    RUNNING = 1
    REFRACTORY = 2


class SaccadeFinder:
    """
    Implements a simple 'on-line' saccade finder similar to that used by the flybratron firmware. 
    """

    def __init__(self, t_start=2.0, threshold=1.0, hysteresis=0.6, duration = 0.02, refractory=1.0):
        """
        Class contructor:

        Parameters
        ----------
        t_start: float
            The time from which to start saccade identification
        threshold: float
            The voltage threshod for identifying a saccade
        hysteresis: float
            The hysteresis used for re-enabling saccade identification given as a 
            precent of the threshold voltage. Saccades renabled if and only if
            abs(voltage) < hysteresis*threshold
        duration: float
            The duration of the saccade stimulus - which would be applied if this
            was a flybratron triggered mode trial.  Use to determine minimum time
            for which saccade detection will be inactive (duration + refratory)
        retractory: float
            The duration of the refractory period - when detection is inactive after
            the triggered mode stimulus. 

        """
        self.t_start = t_start
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.duration = duration
        self.refractory = refractory
        self.reset()


    def reset(self):
        """ Resets the saccade finder """
        self.saccade_ivals = []
        self.saccade_tvals = []
        self.saccade_xvals = []
        self.saccade_svals = []
        self.t_start_running = 0.0
        self.t_start_refractory = 0.0
        self.state = FinderState.REFRACTORY


    def apply(self, tvals, xvals):
        """
        Finds the saccades for the given set of time (tvals) and voltage (xvals). 

        Parameters
        ----------
        tvals: np.array
            1D array of time values. 
        xvals: np.array
            1D array of voltage values.

        Returns
        -------
        dict:  
            A dictionary containing the saccade finder results

            * ``saccade_ivals`` (ndarray): index values of the detected saccades
            * ``sacadde_tvals`` (ndarray): time values of the detected saccades
            * ``saccade_xvals`` (ndarray): voltage values of detected saccades
            * ``saccade_svals`` (ndarray): signs of the detected saccades

        """
        self.reset()
        ivals = np.arange(tvals.size)
        self.t_running = tvals[0] 
        for i, t, x in zip(ivals, tvals, xvals):
            match self.state:
                case FinderState.READY:
                    self.update_on_ready(i,t,x)
                case FinderState.RUNNING:
                    self.update_on_running(i,t,x)
                case FinderState.REFRACTORY:
                    self.update_on_refractory(i,t,x)
                case _:
                    raise RuntimeError("we shouldn't be here")

        rvals = { 
                 'saccade_ivals':  np.array(self.saccade_ivals), 
                 'saccade_tvals':  np.array(self.saccade_tvals), 
                 'saccade_xvals':  np.array(self.saccade_xvals),
                 'saccade_svals':  np.array(self.saccade_svals),
                 }
        return rvals


    def update_on_ready(self, i, t, x):
        """
        Updates the saccade finder using a new data point (t and x value) when
        it is in the ready state.
        """
        if (np.absolute(x) > self.threshold) and (t >= self.t_start):
            self.state = FinderState.RUNNING
            self.t_start_running = t
            self.saccade_ivals.append(i)
            self.saccade_tvals.append(t)
            self.saccade_xvals.append(x)
            self.saccade_svals.append(np.sign(x))

    def update_on_running(self, i, t, x):
        """
        Updates the saccade finder using a new data point (t and x value) when
        it is the running state.
        """
        if t - self.t_start_running > self.duration:
            self.state = FinderState.REFRACTORY
            self.t_start_refractory = t

    def update_on_refractory(self, i, t, x):
        """
        Updates the saccade finder using a new data point (t and x) when it is
        in the the refractory state.
        """
        if t - self.t_start_refractory > self.refractory: 
            if np.absolute(x) < self.threshold*self.hysteresis:
                self.state = FinderState.READY


# Utility functions
# -------------------------------------------------------------------------------

def find_saccades(t, x, param): 
    """
    Applies bandpass filter and then runs the SaccadeFinder.

    Parameters
    ----------
    t: ndarray
        1D array of time values
    x: ndarray
        1D array of delta wingbeat values (typically voltages)
    param: dict
        Dictionary of saccade finder and bandpass filter parameters

        * ``t_start``    (float): time at which saccade detection starts 
        * ``threshold``  (float): threshold used for saccade detection
        * ``hysteresis`` (float): hysteresis used set lower threshold which 
                                  abs(signal) must go below to restart dectection
        * ``duration``   (float): duration of stimulus pulses
        * ``refractory`` (float): duration of refractory period
        * ``hp_fcut``    (float): highpass filter cutoff frequency
        * ``lp_fcut``    (float): lowpass filter cutoff frequency

    Returns
    -------
    dict:  
        A dictionary containing the saccade finder results

        * ``saccade_ivals`` (ndarray): index values of the detected saccades
        * ``sacadde_tvals`` (ndarray): time values of the detected saccades
        * ``saccade_xvals`` (ndarray): voltage values of detected saccades
        * ``saccade_svals`` (ndarray): signs of the detected saccades
        * ``x_bp``          (ndarray): bandpass filtered x values
        
    """
    t_start = param.get('t_start', 0.0) 
    threshold = param.get('threshold', 0.3) 
    hysteresis = param.get('hysteresis', 0.6)
    duration = param.get('duration', 0.01) 
    refractory = param.get('refractory', 1.0)
    hp_fcut = param.get('hp_fcut', 2.0)
    lp_fcut = param.get('lp_fcut', 8.0)

    bp_filter = Bandpass(ival=x[0], hp_fcut=hp_fcut, lp_fcut=lp_fcut)
    x_bp = bp_filter.apply(t, x) 

    saccade_finder = SaccadeFinder(
            t_start = t_start,
            threshold = threshold, 
            hysteresis = hysteresis, 
            duration = duration, 
            refractory = refractory,
            )
    rvals = saccade_finder.apply(t, x_bp)
    rvals['x_bp'] = x_bp
    return rvals


def get_saccade_sections(t, delta_wba_dict, saccade_data, t_window=0.050):
    num_saccade = len(saccade_data['ivals'])
    sections = []
    for n in range(num_saccade):
        i_n = saccade_data['ivals'][n]
        t_n = saccade_data['tvals'][n]
        x_n = saccade_data['xvals'][n]
        s_n = saccade_data['signs'][n]
        mask = np.logical_and(t - t_n > -t_window, t - t_n <= t_window)
        t_mask = t[mask]
        t_mask = t_mask - t_n
        x_orig_mask = delta_wba_dict['orig'][mask]
        x_filt_mask = delta_wba_dict['filt'][mask]
        x_orig_n = delta_wba_dict['orig'][i_n]
        if s_n > 0:
            sections.append({
                't'    : t_mask, 
                'orig' : x_orig_mask-x_orig_n,
                'filt' : x_filt_mask,
                })
        else:
            sections.append({
                't'    : t_mask, 
                'orig' : -(x_orig_mask-x_orig_n),
                'filt' : -x_filt_mask,
                })
    return sections 


def get_mean_saccade_section(sections):

    # Filter sections by length - remove too short/long sections
    num_pts_med = int(np.median([s['t'].size for s in sections]))
    sections_filt = []
    for s in sections:
        num_pts_s = s['t'].size
        if np.absolute(num_pts_med - num_pts_s) < 5:
            sections_filt.append(s)

    # Interpolate sections for taking mean
    min_t = np.array([item['t'][0] for item in sections]).max()
    max_t = np.array([item['t'][-1] for item in sections]).min() 
    t_mean= np.linspace(min_t, max_t, num_pts_med)
    num_sect = len(sections_filt)
    orig = np.zeros((num_sect, num_pts_med))
    filt = np.zeros((num_sect, num_pts_med))
    for i, s in enumerate(sections_filt):
        orig[i,:] = np.interp(t_mean, s['t'], s['orig'])
        filt[i,:] = np.interp(t_mean, s['t'], s['filt'])

    # Find means
    orig_mean = orig.mean(axis=0)
    filt_mean = filt.mean(axis=0)
    orig_mean = np.median(orig, axis=0)
    filt_mean = np.median(filt, axis=0)
    return t_mean, orig_mean, filt_mean




















