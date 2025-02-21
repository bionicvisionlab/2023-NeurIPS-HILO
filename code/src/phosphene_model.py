"""
The phosphene model used was built on top of the open source python library pulse2percept. 
It was eventually implemented in tensorflow, but this version is still required and
is much more user friendly.
"""
import numpy as np
from scipy.stats import multivariate_normal

from pulse2percept.models import AxonMapSpatial, Model
from pulse2percept.implants import ProsthesisSystem, ElectrodeArray
from pulse2percept.stimuli import BiphasicPulseTrain, Stimulus
from pulse2percept.percepts import Percept
from pulse2percept.models.base import NotBuiltError

import pulse2percept as p2p
from collections import OrderedDict

class RectangleImplant(p2p.implants.ProsthesisSystem):
    def __init__(self, x=0, y=0, z=0, rot=0, shape=(15, 15), r=150./2, spacing=400., eye='RE', stim=None,
                 preprocess=True, safe_mode=False):
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.shape = shape
        names = ('A', '1')
        self.earray = p2p.implants.ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z, r=r,
                                    rot=rot, names=names, etype=p2p.implants.DiskElectrode)
        self.stim = stim
        
        # Set left/right eye:
        if not isinstance(eye, str):
            raise TypeError("'eye' must be a string, either 'LE' or 'RE'.")
        if eye != 'LE' and eye != 'RE':
            raise ValueError("'eye' must be either 'LE' or 'RE'.")
        self.eye = eye
        # Unfortunately, in the left eye the labeling of columns is reversed...
        if eye == 'LE':
            # TODO: Would be better to have more flexibility in the naming
            # convention. This is a quick-and-dirty fix:
            names = self.earray.electrode_names
            objects = self.earray.electrode_objects
            names = np.array(names).reshape(self.earray.shape)
            # Reverse column names:
            for row in range(self.earray.shape[0]):
                names[row] = names[row][::-1]
            # Build a new ordered dict:
            electrodes = OrderedDict()
            for name, obj in zip(names.ravel(), objects):
                electrodes.update({name: obj})
            # Assign the new ordered dict to earray:
            self.earray._electrodes = electrodes
    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'shape': self.shape, 'safe_mode': self.safe_mode,
                       'preprocess': self.preprocess})
        return params


class MVGSpatial(AxonMapSpatial):
    """ Multivariate Gaussian Spatial Model

    Parameters
    ----------
    rho : float
        The area of phosphenes in PIXEL coordinates. A phosphene with 
        2xTh, 0.45ms Pdur, 20Hz will always have this area.
        Note that this means the changing the model's xrange and yrange
        will change how large percepts are, because this area is in PIXEL 
        space, and the size of one pixel changes based on the model's grid
    lam : float, [0, .99]
        Eccentricity of phosphenes.
    orient_scale : float near 1
        Corresponds to omega, slope to scale axon orientations by.
    a0-a2 : float
        Coefficients for brightness scaling. Bright = a0*amp^a1 + a2*freq
    a3 : float
        Coefficient for size scaling. Size = a3 * amp * rho
    a4 : float
        Coefficient for eccentricity scaling. Eccentricity = lambda * (pdur / 0.45)^a4

    **params: optional
        Additional params for AxonMapModel. 

        Options:
        --------
        axlambda: double, optional
            Exponential decay constant along the axon(microns).
        rho: double, optional
            Exponential decay constant away from the axon(microns).
        eye: {'RE', LE'}, optional
            Eye for which to generate the axon map.
        xrange : (x_min, x_max), optional
            A tuple indicating the range of x values to simulate (in degrees of
            visual angle). In a right eye, negative x values correspond to the
            temporal retina, and positive x values to the nasal retina. In a left
            eye, the opposite is true.
        yrange : tuple, (y_min, y_max)
            A tuple indicating the range of y values to simulate (in degrees of
            visual angle). Negative y values correspond to the superior retina,
            and positive y values to the inferior retina.
        xystep : int, double, tuple
            Step size for the range of (x,y) values to simulate (in degrees of
            visual angle). For example, to create a grid with x values [0, 0.5, 1]
            use ``x_range=(0, 1)`` and ``xystep=0.5``.
        grid_type : {'rectangular', 'hexagonal'}
            Whether to simulate points on a rectangular or hexagonal grid
        vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
            object that provides ``ret_to_dva`` and ``dva_to_ret`` methods.
            By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
            used.
        n_gray : int, optional
            The number of gray levels to use. If an integer is given, k-means
            clustering is used to compress the color space of the percept into
            ``n_gray`` bins. If None, no compression is performed.
        noise : float or int, optional
            Adds salt-and-pepper noise to each percept frame. An integer will be
            interpreted as the number of pixels to subject to noise in each 
            frame. A float between 0 and 1 will be interpreted as a ratio of 
            pixels to subject to noise in each frame.
        loc_od, loc_od: (x,y), optional
            Location of the optic disc in degrees of visual angle. Note that the
            optic disc in a left eye will be corrected to have a negative x
            coordinate.
        n_axons: int, optional
            Number of axons to generate.
        axons_range: (min, max), optional
            The range of angles(in degrees) at which axons exit the optic disc.
            This corresponds to the range of $\\phi_0$ values used in
            [Jansonius2009]_.
        n_ax_segments: int, optional
            Number of segments an axon is made of.
        ax_segments_range: (min, max), optional
            Lower and upper bounds for the radial position values(polar coords)
            for each axon.
        min_ax_sensitivity: float, optional
            Axon segments whose contribution to brightness is smaller than this
            value will be pruned to improve computational efficiency. Set to a
            value between 0 and 1.
        axon_pickle: str, optional
            File name in which to store precomputed axon maps.
        ignore_pickle: bool, optional
            A flag whether to ignore the pickle file in future calls to
            ``model.build()``.
        n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. 
            Defaults to max number of user CPU cores.
    """

    def __init__(self, **params):
        super(MVGSpatial, self).__init__(**params)
        self.bundles = []
        self.thetas = None

    def get_default_params(self):
        base_params = super(MVGSpatial, self).get_default_params()
        params = {
            # Rho directly corresponds to area
            'rho' : 75, 
            # lam directly corresponds to eccentricity'
            'lam' : 0.9,
            # Thresh_percept is important for this model
            'thresh_percept' : 1/np.exp(1)**2,
            # Scale orientations from axon map model
            'orient_scale' : 1,
            # amp v bright coefficient
            'a0' : 0.4733,
            # amp vs bright exp
            'a1' : 0.5211,
            # freq v bright
            'a2' : 0.016,
            # amp vs size
            'a3' : 0.5,
            # pdur vs ecc exp
            'a4' : -0.2122,
            'amp_cutoff' : 0.25,
            #############
            # For debuging
            'return_thetas' : False
        }
        return {**base_params, **params}

    def _build(self):
        super(MVGSpatial, self)._build()
        self.bundles = self.grow_axon_bundles(prune=True)


    def _scale_amps(self, freq, amp, pdur):
        """.
        """
        return amp

    def _bright(self, freq, amp, pdur):
        # maximum is to match tensorflow variant, where this is needed to preserve gradient flow
        amp = self._scale_amps(freq, amp, pdur)
        return np.where(amp > self.amp_cutoff, self.a0 * np.maximum(amp, 1e-5) ** self.a1 + self.a2 * freq, 0.)

    def _size(self, freq, amp, pdur):
        amp = self._scale_amps(freq, amp, pdur)
        return amp * self.a3

    def _ecc(self, freq, amp, pdur):
        return 0.45**(-self.a4) * pdur**self.a4 

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        if not isinstance(earray, ElectrodeArray):
            raise TypeError("Implant must be of type ElectrodeArray but it is " +
                            str(type(earray)))
        if not isinstance(stim, Stimulus):
            raise TypeError(
                "Stim must be of type Stimulus but it is " + str(type(stim)))
        elec_params = []
        x = []
        y = []
        for e in stim.electrodes:
            try:
                amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
                if amp == 0:
                    continue
                freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
                pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
                x.append(earray[e].x)
                y.append(earray[e].y)
                elec_params.append([freq, amp, pdur])
                
            except KeyError:
                raise TypeError(f"All stimuli must be BiphasicPulseTrains with no " +
                                f"delay dur")
        elec_params = np.array(elec_params, dtype=np.float32)
        ex = np.array(x, dtype=np.float32)
        ey = np.array(y, dtype=np.float32)
        # Actual prediction #
        
        shape = np.array(self.grid.x.shape)
        out = np.zeros(shape, dtype='float32')
        thetas = self.calc_bundle_tangent_fast(ex, ey, bundles=self.bundles) - np.pi/2
        thetas = np.where(thetas < -np.pi/2, thetas + np.pi, thetas)
        thetas = thetas * self.orient_scale
        for x, y, theta, (freq, amp, pdur) in zip(ex, ey, thetas, elec_params):
            rho_prime = np.maximum(self.rho * self._size(freq, amp, pdur), 1)
            lam_prime = np.clip(self.lam * self._ecc(freq, amp, pdur), 0, 0.99)
            bright = self._bright(freq, amp, pdur)
            sy = np.sqrt(rho_prime / (-2*np.pi*np.log(self.thresh_percept)*np.sqrt(1 - lam_prime**2)))
            sx = np.sqrt(rho_prime * np.sqrt(1 - lam_prime**2) / (-2*np.pi*np.log(self.thresh_percept)))
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            exp = 2
            eig = np.array([[sx**exp, 0], [0, sy**exp]])
            cov = R @ eig @ R.T

            # Need to find where the electrode will be in DVA, but in PIXEL coordinates
            center_dva = self.vfmap.ret_to_dva(x, y)
            # build up image using reversed (fliped vertically) image to get orientations right, flip at end
            center_dva = (center_dva[0], -center_dva[1])
            center_pixel = [(center_dva[0] - self.xrange[0])/self.xystep, shape[0] - (center_dva[1] - self.yrange[0])/self.xystep]
            norm = multivariate_normal(mean=center_pixel, cov=cov, allow_singular=True) # centered for now
            def generator_fn(ys, xs, offset=(0, 0)):
                return norm.pdf(np.stack([xs - offset[0], ys-offset[1]], axis=-1)) * 2*np.pi * np.sqrt(np.linalg.det(cov))
            out += bright * np.fromfunction(generator_fn, shape, offset=(0, 0))
        if self.return_thetas:
            self.thetas = thetas
        # out[out<self.thresh_percept] = 0
        return np.flipud(out)
            

    def predict_percept(self, implant, t_percept=None):
        """ Predicts the spatial response
        Override base predict percept to have desired timesteps and 
        remove unneccesary computation

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x 1.
            Will return None if ``implant.stim`` is None.
        """
        # Make sure stimulus is a BiphasicPulseTrain:
        if not isinstance(implant.stim, BiphasicPulseTrain):
            # Could still be a stimulus where each electrode has a biphasic pulse train
            # or a 0 stimulus
            try:
                for i, (ele, params) in enumerate(implant.stim.metadata
                                                ['electrodes'].items()):
                    if (params['type'] != BiphasicPulseTrain or
                            params['metadata']['delay_dur'] != 0) and \
                            np.any(implant.stim[i]):
                        raise TypeError(
                            f"All stimuli must be BiphasicPulseTrains with no " +
                            f"delay dur (Failing electrode: {ele})")
            except KeyError:
                raise TypeError(f"All stimuli must be BiphasicPulseTrains with no " +
                                f"delay dur")
        if isinstance(implant, ProsthesisSystem):
            if implant.eye != self.eye:
                raise ValueError(f"The implant is in {implant.eye} but the model was "
                                 f"built for {self.eye}.")
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, "
                            f"not {type(implant)}.")
        if implant.stim is None:
            return None
        stim = implant.stim
        if t_percept is None:
            n_time = 1
        else:
            n_time = len(t_percept)
        if not np.any(stim.data):
            # Stimulus is 0
            resp = np.zeros(list(self.grid.x.shape) + [n_time],
                            dtype=np.float32)
        else:
            # Make sure stimulus is in proper format
            stim = Stimulus(stim)
            resp = np.zeros(list(self.grid.x.shape) + [n_time])
            # Response goes in first frame
            resp[:, :, 0] = self._predict_spatial(
                implant.earray, stim).reshape(self.grid.x.shape)
        return Percept(resp, space=self.grid, time=t_percept,
                       metadata={'stim': stim.metadata})


class MVGModel(Model):
    """ MultivariateGaussianModel

    Multivariate Gaussian Model
    .. note::
        Using this model in combination with a temporal model is not currently
        supported and will give unexpected results

    Parameters
    ----------
    rho : float
        The area of phosphenes in PIXEL coordinates. A phosphene with 
        2xTh, 0.45ms Pdur, 20Hz will always have this area.
        Note that this means the changing the model's xrange and yrange
        will change how large percepts are, because this area is in PIXEL 
        space, and the size of one pixel changes based on the model's grid
    lam : float, [0, .99]
        Eccentricity of phosphenes.
    orient_scale : float near 1
        Corresponds to omega, slope to scale axon orientations by.
    a0-a2 : float
        Coefficients for brightness scaling. Bright = a0*amp^a1 + a2*freq
    a3 : float
        Coefficient for size scaling. Size = a3 * amp * rho
    a4 : float
        Coefficient for eccentricity scaling. Eccentricity = lambda * (pdur / 0.45)^a4

    **params: optional
        Additional params for AxonMapModel. 

        Options:
        --------
        axlambda: double, optional
            Exponential decay constant along the axon(microns).
        rho: double, optional
            Exponential decay constant away from the axon(microns).
        eye: {'RE', LE'}, optional
            Eye for which to generate the axon map.
        xrange : (x_min, x_max), optional
            A tuple indicating the range of x values to simulate (in degrees of
            visual angle). In a right eye, negative x values correspond to the
            temporal retina, and positive x values to the nasal retina. In a left
            eye, the opposite is true.
        yrange : tuple, (y_min, y_max)
            A tuple indicating the range of y values to simulate (in degrees of
            visual angle). Negative y values correspond to the superior retina,
            and positive y values to the inferior retina.
        xystep : int, double, tuple
            Step size for the range of (x,y) values to simulate (in degrees of
            visual angle). For example, to create a grid with x values [0, 0.5, 1]
            use ``x_range=(0, 1)`` and ``xystep=0.5``.
        grid_type : {'rectangular', 'hexagonal'}
            Whether to simulate points on a rectangular or hexagonal grid
        vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
            object that provides ``ret_to_dva`` and ``dva_to_ret`` methods.
            By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
            used.
        n_gray : int, optional
            The number of gray levels to use. If an integer is given, k-means
            clustering is used to compress the color space of the percept into
            ``n_gray`` bins. If None, no compression is performed.
        noise : float or int, optional
            Adds salt-and-pepper noise to each percept frame. An integer will be
            interpreted as the number of pixels to subject to noise in each 
            frame. A float between 0 and 1 will be interpreted as a ratio of 
            pixels to subject to noise in each frame.
        loc_od, loc_od: (x,y), optional
            Location of the optic disc in degrees of visual angle. Note that the
            optic disc in a left eye will be corrected to have a negative x
            coordinate.
        n_axons: int, optional
            Number of axons to generate.
        axons_range: (min, max), optional
            The range of angles(in degrees) at which axons exit the optic disc.
            This corresponds to the range of $\\phi_0$ values used in
            [Jansonius2009]_.
        n_ax_segments: int, optional
            Number of segments an axon is made of.
        ax_segments_range: (min, max), optional
            Lower and upper bounds for the radial position values(polar coords)
            for each axon.
        min_ax_sensitivity: float, optional
            Axon segments whose contribution to brightness is smaller than this
            value will be pruned to improve computational efficiency. Set to a
            value between 0 and 1.
        axon_pickle: str, optional
            File name in which to store precomputed axon maps.
        ignore_pickle: bool, optional
            A flag whether to ignore the pickle file in future calls to
            ``model.build()``.
        n_threads: int, optional
            Number of CPU threads to use during parallelization using OpenMP. 
            Defaults to max number of user CPU cores.
    """
    def __init__(self, **params):
        super(MVGModel, self).__init__(
            spatial=MVGSpatial(), temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        """Predict a percept
        Overrides base predict percept to keep desired time axes
        .. important::

            You must call ``build`` before calling ``predict_percept``.

        Note: The stimuli should use amplitude as a factor of threshold,
        NOT raw amplitude in microamps

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.
        """
        if not self.is_built:
            raise NotBuiltError("You must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(f"'implant' must be a ProsthesisSystem object, not "
                            f"{type(implant)}.")
        if implant.stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        resp = self.spatial.predict_percept(implant, t_percept=t_percept)
        return resp
