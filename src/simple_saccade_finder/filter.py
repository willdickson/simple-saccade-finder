import numpy as np

class Filter:
    """ Filter base class """

    def __init__(self, ival, fcut):
        """
        Filter Contstructor

        Parameters
        ----------
        ival: float
            initial value
        fcut: float
            cutoff frequency 
        """
        self.ival = ival
        self.fcut = fcut 
        self.reset()

    def reset(self):
        self._val = self.ival

    @property
    def value(self):
        return self._val

    def update(self, x, dt):
        """
        Update the filter on new data point x with timestep dt. 

        Dummy method for base class. 
        """
        self._val = x
        return self._val

    def apply(self, tvals, xvals):
        """
        Applies the filter to the numpy array values (tvals, xvals).

        Paramters
        ---------
        tvals: ndarray
            1D array of time values
        xvals: ndarray 
            1D array of voltage values

        Returns
        -------
        ndarray
            1D array of filtered xvals
        """
        self.ival = xvals[0]
        values = [self.ival]
        for i in range(1,xvals.size):
            dt = tvals[i] - tvals[i-1]
            self.update(xvals[i], dt)
            value = self.value
            values.append(value)
        return np.array(values)


class Lowpass(Filter):
    """ Implements a simple first order lowpass filter """

    def __init__(self, ival=0.0, fcut=1.0):
        """
        Lowpass filter constructor

        Parameters
        ---------
        ival: float
            Initial value for the filter
        fcut: float:
            Lowpass filter cutoff frequency
        """
        super().__init__(ival, fcut)

    def alpha(self, dt):
        numer = 2.0*np.pi*dt*self.fcut
        denom = numer + 1.0
        return numer/denom

    def update(self, x, dt):
        """
        Update the filter on new data point x with timestep dt. 
        """
        alpha = self.alpha(dt)
        self._val = alpha*x + (1.0 - alpha)*self._val
        return self._val


class Highpass(Filter):
    """ Implements a simple first order highpass filter """

    def __init__(self, ival=0.0, fcut=1.0):
        """
        Highpass filter constructor

        Parameters
        ---------
        ival: float
            Initial value for the filter
        fcut: float:
            Highpass filter cutoff frequency
        """
        super().__init__(ival, fcut)

    def reset(self):
        super().reset()
        self._x = self.value 

    def alpha(self,dt):
        return 1.0/(2.0*np.pi*dt*self.fcut + 1.0)

    def update(self, x, dt):
        """
        Update the filter on new data point x with timestep dt. 
        """
        alpha = self.alpha(dt)
        self._val = alpha*self._val + alpha*(x - self._x)
        self._x = x
        return self._val


class Bandpass(Filter):
    """ 
    Implements a bandpass filter consisting of a first order lowpass 
    filter followed by a first order highpass filter. 
    """

    def __init__(self, ival=0.0, hp_fcut=1.0, lp_fcut=6.0):
        """
        Bandpass filter constructor

        Parameters
        ---------
        ival: float
            Initial value for the filter
        fcut: float:
            Bandpass filter cutoff frequency
        """
        self.lp_filter = Lowpass(ival=ival, fcut=lp_fcut)
        self.hp_filter = Highpass(ival=ival, fcut=hp_fcut)
        super().__init__(ival=ival, fcut={'hp': hp_fcut, 'lp': lp_fcut})

    @property
    def value(self):
        return self.hp_filter.value

    def reset(self):
        self.lp_filter.reset()
        self.hp_filter.reset()

    def update(self, x, dt):
        """
        Update the filter on new data point x with timestep dt. 
        """
        return self.hp_filter.update(self.lp_filter.update(x,dt), dt)
