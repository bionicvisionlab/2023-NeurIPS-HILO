import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import zipfile
import pulse2percept as p2p
from pulse2percept.datasets.base import fetch_url

from src.phosphene_model import MVGModel, MVGSpatial

def fetch_dse(model, implant):
    dse_path = 'assets/dse_v2'
    # url = 'https://osf.io/download/646de52cf4be380ba162bc18/'
    # updated model (corrects small bug in patient parameter a4 of previous model)
    url = 'https://osf.io/download/6555798c062a3e1162ee0412/' 
    if not (os.path.exists(dse_path) and os.path.isdir(dse_path) and 
            os.path.exists(os.path.join(dse_path, 'saved_model.pb'))):
        if not os.path.exists(dse_path + '.zip'):
            fetch_url(url, dse_path + '.zip')
        if not os.path.exists(dse_path):
            os.mkdir(dse_path)
        with zipfile.ZipFile(dse_path + '.zip', 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dse_path))

    return load_model(dse_path, model, implant)
        

def load_model(path, model, implant):
    nn1 = tf.keras.models.load_model(path)
    def clone_fn(layer): 
        # isinstance raises bug with some layers and nonetype
        if 'UniversalMVGLayer' in str(type(layer)):
            return UniversalMVGLayer(model, implant)
            
        return layer.__class__.from_config(layer.get_config())
    
    nn = tf.keras.models.clone_model(nn1, clone_function=clone_fn)
    nn.set_weights(nn1.get_weights())
    nn.compile(loss='mse', metrics=[])
    del nn1
    return nn



def load_mnist(model, scale=2.0, pad=2):
    """ Loads mnist targets, rescaled and padded to the model's output shape """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255 * scale
    x_train = tf.image.resize_with_pad(x_train.reshape((-1, 28, 28, 1)), model.grid.shape[0]-2*pad, model.grid.shape[1]-2*pad, antialias=True)[:, :, :, 0]
    x_train = np.pad(x_train, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    x_train = x_train.reshape((-1, model.grid.shape[0], model.grid.shape[1], 1))
    x_test = x_test.astype('float32') / 255 * scale
    x_test = tf.image.resize_with_pad(x_test.reshape((-1, 28, 28, 1)), model.grid.shape[0]-2*pad, model.grid.shape[1]-2*pad, antialias=True)[:, :, :, 0]
    x_test = np.pad(x_test, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    x_test = x_test.reshape((-1, model.grid.shape[0], model.grid.shape[1], 1))

    return (x_train, y_train), (x_test, y_test)


class UniversalMVGLayer(tf.keras.layers.Layer):
    def __init__(self, p2pmodel, implant, activity_regularizer=None, amp_cutoff=0.25, n_interp=200, threshold_miss=None, **kwargs):
        """ Predicts percepts from (batched) input stimuli
        Universal signifies that it can predict percepts for varying patient-specific parameters without needing recompiling

        Parameters:
        -----------
        p2pmodel : MVGModel from pulse2percept
            Used for axons and simulated range
        implant  :  p2p.implant. 
            Only the shape from this will be used, so that all electrode locations can be 
            specified just by elec_x and elec_y.
            This should be at (0, 0). If not, then all patient specific parameters will be offsets
            from wherever the passed implant is located.
        amp_cutoff : float
            Threshold beneath which amplitudes will not be visible (0.25 is standard value)
        n_interp : int
            Number of interpolation points to use for interpolating axon slopes at new points.
            with 200, the typical error in axon is 1e-10, except at the raphe. If an electrode
            is exactly on the raphe, then the interpolation does not match p2p MVGModel, but real
            axon trajectories are quite variable right along the raphe anyways, so this is okay.
        threshold_miss : None or float
            If passed, the level of threshold misspecification used.
        """
        super(UniversalMVGLayer, self).__init__(trainable=False, 
                                                   name="UniversalMVG", 
                                                   dtype='float32',
                                                   activity_regularizer=activity_regularizer, **kwargs)

        if not (isinstance(p2pmodel, MVGModel) or isinstance(p2pmodel, MVGSpatial)):
            raise ValueError("Must pass in a valide MVGModel")
        if not isinstance(implant, p2p.implants.ProsthesisSystem):
            raise ValueError("Invalid implant")

        # self.p2pmodel = p2pmodel
        # self.implant = implant
        self.xrange = tf.constant(p2pmodel.xrange, dtype='float32')
        self.yrange = tf.constant(p2pmodel.yrange, dtype='float32')
        self.xystep = tf.constant(p2pmodel.xystep, dtype='float32')
        self.percept_shape = tf.constant(p2pmodel.grid.shape)
        self.pixelgrid = tf.constant(np.swapaxes(np.indices(p2pmodel.grid.shape), 0, 2).reshape(-1, 2), dtype='float32')
        self.npoints = tf.constant(self.pixelgrid.shape[0])
        self.thresh_percept = tf.constant(1 / tf.exp(1.)**2, dtype='float32')
        x, y = p2pmodel.retinotopy.dva2ret(p2pmodel.loc_od[0], p2pmodel.loc_od[1]) 
        self.od_off_x = tf.constant(x, dtype='float32')
        self.od_off_y = tf.constant(y, dtype='float32')

        # # for true method
        self.bundles = p2pmodel.grow_axon_bundles(prune=True)
        self.flat_bundles = tf.concat(self.bundles, axis=0)
        axon_idx = [[idx] * len(ax) for idx, ax in enumerate(self.bundles)]
        axon_idx = [item for sublist in axon_idx for item in sublist]
        self.axon_idx = tf.constant(axon_idx)
        
        # for interp method
        self.n_interp = n_interp
        # retinal coords
        self.xlow = tf.constant(np.min(p2pmodel.grid.xret), dtype='float32')
        self.xhigh = tf.constant(np.max(p2pmodel.grid.xret), dtype='float32')
        self.ylow = tf.constant(np.min(p2pmodel.grid.yret), dtype='float32')
        self.yhigh = tf.constant(np.max(p2pmodel.grid.yret), dtype='float32')
        self.interp_step_x = tf.constant((self.xhigh-self.xlow) / (self.n_interp - 1), dtype='float32')
        self.interp_step_y = tf.constant((self.yhigh-self.ylow) / (self.n_interp - 1), dtype='float32')
        # (n_interp, n_interp, 2)
        self.grid_axis_x = tf.linspace(self.xlow, self.xhigh, self.n_interp)
        self.grid_axis_y = tf.linspace(self.ylow, self.yhigh, self.n_interp)
        gridx, gridy = tf.meshgrid(self.grid_axis_x, self.grid_axis_y)
        self.slopes = tf.constant(p2pmodel.calc_bundle_tangent_fast(gridx, gridy, bundles=p2pmodel.bundles), dtype='float32') - np.pi/2
        
        # Get implant parameters
        self.n_elecs = tf.constant(len(implant.electrodes))
        self.elec_x = tf.reshape(tf.constant([implant[e].x for e in implant.electrodes], dtype='float32'), (1, -1))
        self.elec_y = tf.reshape(tf.constant([implant[e].y for e in implant.electrodes], dtype='float32'), (1, -1))
        self.pi = tf.constant(np.pi, dtype='float32')
        self.amp_cutoff = tf.constant(amp_cutoff, dtype='float32')
        # threshold misspecification
        if threshold_miss is not None:
            signs = np.array(np.random.randint(0, 2, size=(len(implant.electrodes)))*2 - 1, dtype='float32')
            self.threshold_miss = tf.constant( (np.random.rand(len(implant.electrodes)).astype('float32') * threshold_miss  + 1)**signs, dtype='float32')
        else:
            self.threshold_miss = None


    def compute_output_shape(self, input_shape):
        batched_percept_shape = tuple([input_shape[0]] + list(self.percept_shape))
        return batched_percept_shape

    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # print('recompiling')
        """ inputs should be a list, 
        first item is n_elecs x 3 array containing freqs, amps, and pdurs for each electrode in the array
        second item is [rho, axlambda, a0-a4, implant_x, implant_y, rot, loc_od_x, loc_od_y]"""

        freq = inputs[0][:, :, 0]
        amp = inputs[0][:, :, 1]

        # this will get compiled out but it never changes within a model anyways
        if self.threshold_miss is not None:
            amp = amp / self.threshold_miss[None, :]


        pdur = inputs[0][:, :, 2]

        # all subjectparams are (batch, 1)
        rho = inputs[1][:, 0][:, None]
        lam = inputs[1][:, 1][:, None]
        orient_scale = inputs[1][:, 2][:, None]
        a0 = inputs[1][:, 3][:, None]
        a1 = inputs[1][:, 4][:, None]
        a2 = inputs[1][:, 5][:, None]
        a3 = inputs[1][:, 6][:, None]
        a4 = inputs[1][:, 7][:, None]
        
        implant_x = inputs[1][:, 8][:, None]
        implant_y = inputs[1][:, 9][:, None]
        rot = inputs[1][:, 10][:, None]
        
        # In RETINAL coords
        loc_od_x = inputs[1][:, 11][:, None]
        loc_od_y = inputs[1][:, 12][:, None]

        # (1, elecs)
        ex = self.elec_x
        ey = self.elec_y
        # apply rotation and translation
        ex = ex * tf.cos(rot) - ey * tf.sin(rot) + implant_x
        ey = self.elec_x * tf.sin(rot) + ey * tf.cos(rot) + implant_y

        # thetas = tf.constant(self.p2pmodel.calc_bundle_tangent_fast(ex, ey, bundles=self.p2pmodel.bundles)) - self.pi/2
        offx, _ = loc_od_x - self.od_off_x, loc_od_y - self.od_off_y
        
        thetas = self.interp_bundle_tangent(ex-offx, ey)
        # thetas = self.calc_bundle_tangent(ex-offx, ey) - np.pi/2
        thetas = tf.where(thetas < -np.pi/2, thetas + np.pi, thetas)
        thetas = thetas * orient_scale # (batch, elecs)
        # self.thetas = thetas
        # all (batch, elecs)

        rho_scaled = tf.maximum(rho * amp * a3, 1) # dont allow images with less than 1 pixel. If something should be gone, then bright will take care of it
        lam_scaled = tf.clip_by_value(lam *  0.45**(-a4)*pdur**a4, 0, 0.99)
        

        # avoid inf gradient using maximum. Output still 0
        # bright_scaled = a0 * amp ** a1 + a2 * freq
        bright_scaled = tf.where(amp > self.amp_cutoff, a0 * tf.maximum(amp, 1e-5) ** a1 + a2 * freq, 0.)
        temp1 = -2*self.pi*tf.math.log(self.thresh_percept)
        temp2 = tf.sqrt(1 - lam_scaled**2)
        # eigenvalues^2 of covariance matrix
        # (batch, elecs)
        sy = rho_scaled / (temp1 * temp2)
        sx = rho_scaled * temp2 / temp1
        # rotation of covariance matrix
        # (batch, elecs)
        sintheta = tf.sin(thetas)
        costheta = tf.cos(thetas)
        # R is (batch, elecs, 2, 2)

    
        R = tf.stack([tf.stack([costheta, -sintheta], axis=-1), 
                      tf.stack([sintheta, costheta], axis=-1)], axis=-2)
        # (batch, elecs, 2, 2)
        eig = tf.linalg.diag(tf.stack([sx, sy], axis=-1))
        # (batch, elecs, 2, 2)
        cov = R @ eig @ tf.transpose(R, perm=[0, 1, 3, 2])
        
        # 2*self.pi*tf.sqrt(tf.linalg.det(cov))
        # calculate determinant manually, fixes JIT bug and much faster
        dets = cov[..., 0, 0]*cov[..., 1, 1] - cov[..., 1, 0]*cov[..., 0, 1]
        # unnormalize = 2*self.pi*tf.sqrt(dets)
        covinv_temp = tf.stack([tf.stack([cov[..., 1, 1], -cov[..., 0, 1]], axis=-1), 
                                    tf.stack([-cov[..., 1, 0], cov[..., 0, 0]], axis=-1)], axis=-2)
        covinv = 1/dets[..., None, None] * covinv_temp
    
        # need to find where each electrode will be in PIXEL coordinates
        # all (batch, elecs)
        center_dva = self.ret2dva(ex, ey)
        center_x = (center_dva[0] - self.xrange[0]) / self.xystep
        # build it up upside down so that once its flipped, the orientation will be correct
        center_y = tf.cast(self.percept_shape[0], 'float32') - (center_dva[1] - self.yrange[0])/self.xystep
        # (1, elecs, 2)
        center_pixels = tf.stack([center_x, center_y], axis=-1)
        # center_pixels = tf.repeat(center_pixels, tf.keras.backend.shape(cov)[0], axis=0)

        # return center_pixels, cov, covinv, bright_scaled
        # pixelgrid is npoints, 2
        t1 = self.pixelgrid[:, None, None, :] - center_pixels
        
        # t3 = tf.matmul(t1[..., None, :] @ covinv, t1[..., None, :], transpose_b=True)[..., 0, 0]
        # covinv is symmetric
        t3 = (t1[..., 0]**2 * covinv[..., 0, 0] +  2*t1[..., 0]*t1[..., 1]*covinv[..., 1, 0] + 
              t1[..., 1]**2*covinv[..., 1, 1] )
        
        imgs = tf.reduce_sum(tf.exp(-t3/2) * bright_scaled, axis=-1)
        intensities = tf.image.flip_up_down(tf.reshape(tf.transpose(imgs, perm=[1, 0]), (-1, self.percept_shape[0], self.percept_shape[1]))[..., None])[..., 0]
        return intensities

    @tf.function(jit_compile=True)
    def calc_bundle_tangent(self, x, y):
        # for each x, y, find closest point in self.flat_bundles
        # (batch, elecs, points)
        d2 = (x[..., None] - self.flat_bundles[None, None, :, 0])**2 + (y[..., None] - self.flat_bundles[None, None, :, 1])**2
        # (batch, elecs)
        min_idx = tf.argmin(d2, axis=-1, output_type=tf.int32)
        # (batch, elecs)
        segs = tf.gather(self.axon_idx, min_idx)
        prev_segs = tf.gather(self.axon_idx, min_idx-1)
        next_segs = tf.gather(self.axon_idx, min_idx+1)
        offset_l = tf.where(prev_segs == segs, -1, 0)
        offset_r = tf.where(next_segs == segs, 1, 0)
        dx = tf.gather(self.flat_bundles, min_idx + offset_r) - tf.gather(self.flat_bundles, min_idx + offset_l)
        dx = tf.stack([dx[..., 0], dx[..., 1] * -1], axis=-1)
        tangent = tf.math.atan2(dx[..., 1], dx[..., 0])
        # Confine to (-pi/2, pi/2):
        tangent = tf.where(tangent < -self.pi/2, tangent+self.pi, tangent)
        tangent = tf.where(tangent > self.pi/2, tangent - self.pi, tangent)
        return tangent
    
    @tf.function(jit_compile=True)
    def interp_bundle_tangent(self, x, y):
        query_x = (x - self.xlow) / self.interp_step_x
        query_y = (y - self.ylow) / self.interp_step_y
        query = tf.stack([query_x, query_y], axis=-1)
        grid = tf.repeat(self.slopes[None, :, :, None], tf.keras.backend.shape(query)[0], axis=0)
        return tfa.image.interpolate_bilinear(grid, query , indexing='xy')[..., 0]
        
    def ret2dva(self, x, y):
        """Converts retinal distances (um) to visual angles (deg)

        This function converts an eccentricity measurement on the retinal
        surface(in micrometers), measured from the optic axis, into degrees
        of visual angle using Eq. A6 in [Watson2014]_.

        Parameters
        ----------
        x, y : double or array-like
            Original x and y coordinates on the retina (microns)

        Returns
        -------
        x_dva, y_dva : double or array-like
            Transformed x and y coordinates (degrees of visual angle, dva)
        """
        phi_um, r_um = self.cart2pol(x, y)
        sign = tf.math.sign(r_um)
        r_mm = 1e-3 * tf.abs(r_um)
        r_deg = 3.556 * r_mm + 0.05993 * r_mm ** 2 - 0.007358 * r_mm ** 3
        r_deg += 3.027e-4 * r_mm ** 4
        r_deg *= sign
        return self.pol2cart(phi_um, r_deg)
    
    def cart2pol(self, x, y):
        rho = tf.sqrt(x**2 + y**2)
        theta = tf.math.atan2(y, x)
        return theta, rho

    def pol2cart(self, theta, rho):
        x = rho * tf.cos(theta)
        y = rho * tf.sin(theta)
        return x, y
    

default_phi_ranges = {
        'rho' : [10, 200],
        'lam' : [0.72, 0.98],
        'orient_scale' : [0.9, 1.1],
        'a0' : [.27, .57],
        'a1' : [.42, .62],
        'a2' : [0.005, 0.025],
        'a3' : [.2, .7],
        'a4' : [-0.5, -0.1],
        'implant_x' : [-500, 500],
        'implant_y' : [-500, 500],
        'implant_rot' : [-np.pi/6, np.pi/6],
        'loc_od_x' : [3700, 4700],
        'loc_od_y' : [0, 1000]
    }
def rand_model_params(n, params=None, ranges={}):
    """ Helper function to get random phi (patient specific parameters) """
    ood = False
    default_ranges = {
        'rho' : [10, 200],
        'lam' : [0.72, 0.98],
        'orient_scale' : [0.9, 1.1],
        'a0' : [.27, .57],
        'a1' : [.42, .62],
        'a2' : [0.005, 0.025],
        'a3' : [.2, .7],
        'a4' : [-.5, -.1], 
        'implant_x' : [-500, 500],
        'implant_y' : [-500, 500],
        'implant_rot' : [-np.pi/6, np.pi/6],
        'loc_od_x' : [3700, 4700],
        'loc_od_y' : [0, 1000]
    }

    if params is None or params == 'all':
        params = default_ranges.keys()
    
    new_ranges = {}
    for param in params:
        if param in ranges.keys():
            new_ranges[param] = ranges[param]
        else:
            new_ranges[param] = default_ranges[param]
    
    out = np.zeros((n, len(params)))

    for i, param in enumerate(params):
        row = new_ranges[param][0] + (new_ranges[param][1] - new_ranges[param][0]) * np.random.rand(n)
        if not ood:
            out[:, i] = np.clip(row, new_ranges[param][0], new_ranges[param][1])
    return out