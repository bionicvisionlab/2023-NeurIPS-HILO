

from src.phosphene_model import RectangleImplant, MVGModel
from src.DSE import UniversalMVGLayer, load_model, rand_model_params, default_phi_ranges, NaiveEncoding, MODEL_NAMES_TO_VERSION_OSF

import pulse2percept as p2p
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import matlab
import matlab.engine
import io


def patient_from_phi(phi, model, implant, implant_kwargs={}, layer=None):
    """
    Util to update/create a model and implant for a specific patient's phi parameters
    Phi must be a DICT (see phi_arr2dict)
    """
    rebuild_attrs = ['loc_od', 'loc_od_x', 'loc_od_y']
    # prepare the implant
    implant_kwargs_phi = {k[8:]:v for k, v in phi.items() if k[:8] == 'implant_'}
    implant_kwargs |= implant_kwargs_phi # overwrites any copies with those in phi
    implant_new = implant.__class__(**implant_kwargs)

    # prepare the model
    rebuild = False
    for attr, val in phi.items():
        if attr[:8] == 'implant_':
            continue
        if attr in rebuild_attrs:
            rebuild = True
        setattr(model, attr, val)
        if layer is not None:
            setattr(layer, attr, val)
    if rebuild:
        model.build()
        if layer is not None:
            layer.__init__(model, implant)

    return model, implant_new


def phi_arr2dict(phi, names=None, vfmap=p2p.topography.Watson2014Map()):
    """
    Pretty dict form of phi
    Some values are changed to match how p2p expects them.
    All values the same except loc_od, which is 
    converted to DVA for the dict, and implant_rot, which is converted to degrees
    """
    if names is None:
        names = ['rho', 'lam', 'orient_scale', 'a0', 'a1', 'a2', 'a3', 'a4', 
                'implant_x', 'implant_y', 'implant_rot', 'loc_od_x', 'loc_od_y'][:phi.shape[-1]]
        
    # always need both loc_od
    loc_od = None
    if 'loc_od_x' in names and 'loc_od_y' in names:
        loc_od = (phi[names.index('loc_od_x')], phi[names.index('loc_od_y')])
    elif 'loc_od_x' in names:
        loc_od = (phi[names.index('loc_od_x')], 406.97)
    elif 'loc_od_y' in names:
        loc_od = (4205.404, phi[names.index('loc_od_y')])
    loc_od = vfmap.ret_to_dva(loc_od[0], loc_od[1])

    phi_dict = {}
    for idx, name in enumerate(names):
        if 'loc_od' in name:
            phi_dict['loc_od'] = loc_od
        elif name == 'implant_rot':
            phi_dict[name] = np.rad2deg(phi[idx])
        else:
            phi_dict[name] = phi[idx]

    return phi_dict


def patient_from_phi_arr(phi, model, implant, implant_kwargs={}, phi_names=None):
    """ Helper util to generate a phosphene model from a patient array """
    phi_dict = phi_arr2dict(phi, names=phi_names)
    return patient_from_phi(phi_dict, model, implant, implant_kwargs=implant_kwargs)


class HILOPatient():
    """Interfaces with HNA models"""
    def __init__(self, model=None, implant=None,  implant_kwargs=None, phi_true=None, metrics=None,
                 phi_names=None, comp=True, loss='mse', matlab_dir=None, nopt=2, noise=0.001,
                 misspecification=None, miss_options=[],
                 kernel='Matern52', acquisition='MUC', ranges=None, maxiter=100,
                 use_phi_names=None, dse=None, version='v2'):
        """

        Model and Implant should correspond to the phi you want to optimize!

        A bit overloaded right now

        Parameters:
        -----------
        model : p2p.models.Model
            The model to optimize, with defaults set. 
        implant : p2p.implants.ProsthesisSystem
            The implant for which to optimize stimuli.
            Note that just the class is used, any defaults should be specified in implant_kwargs
        implant_kwargs : dict
            Dict of kwargs to pass to implant (since implant is not stateful) 
        """
        self.version = MODEL_NAMES_TO_VERSION_OSF[version][0]
        self.universal_mvg_kwargs = MODEL_NAMES_TO_VERSION_OSF[version][2]
        self.model = model
        if self.model is None:
            self.model = MVGModel(xrange=(-12, 12), yrange=(-12, 12), xystep=0.5, rho=250/4)
        self.model.build()

        self.implant = implant
        if self.implant is None:
            self.implant = RectangleImplant()
        # now pass in kwargs
        if implant_kwargs is None:
            implant_kwargs = {'spacing' : 400, 'shape' : (15, 15)}
        self.implant_kwargs = implant_kwargs
        self.implant = self.implant.__class__(**self.implant_kwargs)

        self.dse = dse
        if isinstance(self.dse, str):
            dse_nn = load_model(self.inverse_method)
        elif isinstance(self.dse, tf.keras.Model):
            dse_nn = self.dse

        self.phi_true = phi_true
        self.phi_names = phi_names
        if self.phi_names is None:
            self.phi_names = list(default_phi_ranges[self.version].keys())
        if self.phi_true is None:
            self.phi_true = [model.rho, model.lam, model.orient_scale, model.a0, model.a1, model.a2, model.a3, model.a4]
            for attr in ['x', 'y', 'rot']:
                if attr in self.implant_kwargs:
                    self.phi_true.append(self.implant_kwargs[attr])
                else:
                    self.phi_true.append(0.)
            # loc od needs to be in RET
            loc_od = model.vfmap.dva_to_ret(model.loc_od[0], model.loc_od[1])
            self.phi_true += [loc_od[0], loc_od[1]]
        self.phi_true = tf.constant(self.phi_true, dtype='float32')

        self.loss_type = loss
        # hacky but easy
        if not np.any([i in str(dse_nn.layers) for i in ['UniversalMVGLayer', 'MVGLayer']]):
            nn_stims = dse_nn
        else:
            # need to extract stimulus part
            layernames = [l.name for l in dse_nn.layers]
            if 'stims' not in layernames:
                raise ValueError ("Network must have a stimuli layer named 'stims'")
            stimlayer = dse_nn.layers[layernames.index('stims')]

            # stimulus encoder to be used
            nn_stims = tf.keras.Model(inputs=dse_nn.inputs, outputs=stimlayer.output)
            # patients TRUE perceptual model
            self.decoder_layer = self.get_decoder_layer(dse_nn.layers[-1], misspecification, miss_options, self.universal_mvg_kwargs)

        # construct a mismatch_dse, where the target is encoded tp stimulus with the provided phi, but the phosphene
        # is decoded from the stimulus using the patient's ground truth phi
        inp_img = tf.keras.layers.Input(shape=nn_stims.inputs[0].get_shape()[1:])
        inp_phi = tf.keras.layers.Input(shape=nn_stims.inputs[1].get_shape()[1:])
        stims = nn_stims([inp_img, inp_phi])
        out = self.decoder_layer([stims, tf.broadcast_to(self.phi_true[None, ...], tf.keras.backend.shape(inp_phi))])
        self.mismatch_dse = tf.keras.Model(inputs=[inp_img, inp_phi], outputs=out)
        self.mismatch_dse.compile(loss=loss, metrics=metrics)

        if comp:
            self.mismatch_dse.call = tf.function(self.mismatch_dse.call, jit_compile=True)

        # construct naive nn
        inp_img = tf.keras.layers.Input(shape=nn_stims.inputs[0].get_shape()[1:])
        inp_phi = tf.keras.layers.Input(shape=nn_stims.inputs[1].get_shape()[1:])
        self.naive_layer = NaiveEncoding(self.implant, freq=20, stimrange=(0, 6))
        stims = self.naive_layer(inp_img)
        out = self.decoder_layer([stims, tf.broadcast_to(self.phi_true[None, ...], tf.keras.backend.shape(inp_phi))])
        self.nn_naive = tf.keras.Model(inputs=[inp_img, inp_phi], outputs=out)
        self.nn_naive.compile(loss=loss, metrics=metrics)

        # HILO STUFF
        self.noise = noise
        self.hiloModel = None
        self.use_phi_names = use_phi_names
        if self.use_phi_names is None:
            self.use_phi_names = self.phi_names
        if ranges is None:
            ranges = default_phi_ranges[self.version]
        self.ranges = ranges
        thetacov = np.array([[0.54], [3.63]], dtype='double')
        if matlab_dir is not None:
            self.setup_matlab(matlab_dir, kernel=kernel, ranges=ranges, thetacov=thetacov, acquisition=acquisition, maxiter=maxiter, nopt=nopt)
        

    def get_decoder_layer(self, decoder_layer, misspecification, miss_options, universalmvg_kwargs={}):
        """ Allows for the decoder layer to be misspecified"""
        self.missspecification = misspecification
        self.miss_options = miss_options
        if misspecification == 'beta':
            t = np.random.randint(0, 4)
            beta_opts = np.array([[-1.3, -1.3, -2.5, -2.5], [0.1, 1.3, 0.1, 1.3]])
            beta_sup, beta_inf = beta_opts[:, t]
            model = p2p.models.MVGModel(xrange=(-12, 12), yrange=(-12, 12), xystep=0.5, beta_sup=beta_sup, beta_inf=beta_inf).build()
            implant = RectangleImplant(spacing=400, shape=(15, 15))
            return UniversalMVGLayer(model, implant, **universalmvg_kwargs)
        elif misspecification == 'thresh':
            perc_change = float(miss_options[0])
            model = p2p.models.MVGModel(xrange=(-12, 12), yrange=(-12, 12), xystep=0.5).build()
            implant = RectangleImplant(spacing=400, shape=(15, 15))
            return UniversalMVGLayer(model, implant, threshold_miss=perc_change, **universalmvg_kwargs)
        else:
            return decoder_layer


    ################## HILO Matlab interface functions ###################################
    def setup_matlab(self, matlab_dir, kernel='Matern52', ranges=None, thetacov=None,
                      acquisition='MUC', maxiter=200, nopt=3):
        """
        Sets up the interface to matlab
        Returns nothing, but after calling, the hilo patient will be set up to 
        do PBO with the ranges and names sent.
        See setup_hilo.m for details and more default values
        """
        if matlab_dir is None:
            raise ValueError("Must supply a matlabdir where setup_hilo scripts are")
        if thetacov is None:
            thetacov = np.array([0., 0.], dtype='double')

        self.matlab_dir = matlab_dir
        self.kernel = kernel
        self.ub = np.array([[v[1]] for k, v in ranges.items() if k in self.use_phi_names], dtype='double')
        self.lb = np.array([[v[0]] for k, v in ranges.items() if k in self.use_phi_names], dtype='double')
        if self.missspecification == 'OOD' and len(self.miss_options) > 1 and self.miss_options[1] == 'adjust':
            ub = []
            lb = []
            for k, v in ranges.items():
                if k not in self.use_phi_names:
                    continue
                m = (v[1] + v[0]) / 2
                r = v[1] - v[0]
                ub.append([m + r/2 + r*float(self.miss_options[0])/2])
                lb.append([m- r/2 - r*float(self.miss_options[0])/2])
            self.lb = np.array(lb, dtype='double')
            self.ub = np.array(ub, dtype='double')

        self.use_phi_indices = np.array([list(ranges.keys()).index(i) for i in self.use_phi_names])
        self.acquisition = acquisition
        self.maxiter = maxiter
        self.nopt = nopt
        self.thetacov = np.array(np.array(thetacov).reshape((1, -1)), dtype='double')

        self.eng = matlab.engine.start_matlab()
        self.eng.cd(matlab_dir, nargout=0)

        self.hiloModel = self.eng.setup_hilo(self.kernel, self.ub, self.lb, self.thetacov, self.acquisition, self.maxiter, self.nopt)
        self.iter = 1
        self.d = len(self.ub)
        self.out = io.StringIO()
        self.err = io.StringIO()


    def reset_hilo(self):
        self.hiloModel = self.eng.setup_hilo(self.kernel, self.ub, self.lb, self.thetacov, self.acquisition, self.maxiter, self.nopt)
        self.iter = 1


    def verify_hilo(self, xtrain, ctrain):
        if self.hiloModel is None:
            raise ValueError("Hilo model not set up")
        ind = np.concatenate([self.use_phi_indices, len(default_phi_ranges[self.version]) + self.use_phi_indices])
        xtrain = xtrain[ind, ...]
        if xtrain.shape[0] != self.d * 2:
            raise ValueError('Xtrain must be of shape (nd*2, npoints)')
        if len(ctrain.shape) == 1:
            ctrain = ctrain.reshape((1, -1))
        if xtrain.shape[1] != ctrain.shape[1]:
            raise ValueError('Number of datapoints in xtrain and ctrain does not match')
        
        if not xtrain.data.c_contiguous or xtrain.dtype != 'double':
            xtrain = np.array(xtrain, dtype='double')
        if not ctrain.data.c_contiguous or ctrain.dtype != 'double':
            ctrain = np.array(ctrain, dtype='double')
        return xtrain, ctrain


    def add_phi(self, phi, double=False):
        if double:
            # its a stacked (duel) of two phis
            out = np.zeros((2*len(self.phi_names)), dtype='double')
            out[np.concatenate([self.use_phi_indices, self.use_phi_indices + len(self.phi_names)])] = phi
            for idx, param in enumerate(self.phi_names):
                if param in self.use_phi_names:
                    continue
                out[idx] = (self.ranges[param][1] + self.ranges[param][0]) / 2
                out[idx + len(self.phi_names)] = (self.ranges[param][1] + self.ranges[param][0]) / 2
            return out
        out = np.zeros((len(self.phi_names)), dtype='double')
        out[self.use_phi_indices] = phi
        for idx, param in enumerate(self.phi_names):
            if param in self.use_phi_names:
                continue
            out[idx] = (self.ranges[param][1] + self.ranges[param][0]) / 2
        return out

    def hilo_acquisition(self, xtrain, ctrain):
        """Returns an array of shape (d*2)"""
        if xtrain is None :
            newx = self.eng.acquisition(self.hiloModel, [], [], self.iter, stdout=self.out,stderr=self.err)
        else:
            xtrain, ctrain = self.verify_hilo(xtrain, ctrain)
            newx = self.eng.acquisition(self.hiloModel, xtrain, ctrain, self.iter, stdout=self.out,stderr=self.err)
        self.iter += 1
        newx =  np.array(newx, dtype='double').squeeze()
        newx = self.add_phi(newx, double=True)
        return newx
    
    def hilo_update_posterior(self, xtrain, ctrain):
        xtrain, ctrain = self.verify_hilo(xtrain, ctrain)
        # print(ctrain)
        self.hiloModel = self.eng.update_posterior(self.hiloModel, xtrain, ctrain, stdout=self.out,stderr=self.err)

    def hilo_identify_best(self, xtrain, ctrain):
        xtrain, ctrain = self.verify_hilo(xtrain, ctrain)
        bestx = self.eng.identify_best(self.hiloModel, xtrain, ctrain, stdout=self.out,stderr=self.err)
        # need to convert bestx back to full dimensions
        bestx = np.array(bestx, dtype='double').squeeze()
        bestx = self.add_phi(bestx, double=False)
        return bestx
        
    ################## END HILO Matlab interface functions ###################################

    def flag_suspicious(self, logs, stims):
        """Return true if log entry is suspicious (did not converge)"""
        for i in range(stims.shape[0]):
            stim = stims[i]
            if tf.experimental.numpy.allclose(stim[:, 0], 0, atol=1e-4):
                return "zero_freq"
            if tf.experimental.numpy.allclose(stim[:, 1], 0, atol=1e-4):
                return "zero_amp"
            if tf.experimental.numpy.allclose(stim[:, 2], 0, atol=1e-4):
                return "zero_pdur"
            
        last_epoch = logs['last_epoch']
        es_steps = self.descent_opt_kwargs['es_patience'] if 'es_patience' in self.descent_opt_kwargs.keys() else 100
        if self.descent_nsteps - last_epoch < es_steps:
            return "no_es"
        if last_epoch < self.warmup_steps + es_steps + 250:
            return "stopped_early"
        if last_epoch > 6000:
            return "long"
        
        if logs['best_loss'] > self.loss_threshold:
            return 'high_loss'

        return "false"

    def inv_target_dse(self, target, phi):
        if len(target.shape)  == 3:
            target = target[None, ...]
        if len(phi.shape) == 1:
            phi = phi[None, ...]

        logs = {}
        start = time.time()
        percept = self.mismatch_dse([target, phi])
        logs['time'] = time.time() - start

        # returning stimulus currently not supported with this method (for speed reasons)
        return None, percept, logs


    def inv_target(self, target, phi):
        return self.inv_target_dse(target, phi)


    def simulate_decision(self, loss1, loss2):
        """
        Stochastic Bernouli sampling based on distribution of differences
        1 = loss1, 0=loss2 (matches Fauvel 2020)
        """
        thresh = 1 / (1 + np.exp(-(loss2-loss1)/self.noise))
        return int(random.random() < thresh)


    def loss(self, yt, yp, fn='mse'):
        if fn == 'mse':
            return tf.reduce_mean((yt-yp)**2)
        elif fn == 'mae':
            return tf.reduce_mean(tf.math.abs(yt - yp))
        elif callable(fn):
            return tf.reduce_mean(fn(yt, yp))
        raise ValueError(f"Unknown loss fn {fn}")


    def run_mismatch(self, targets, phi):
        return self.mismatch_dse([targets, phi])

    def duel(self, target, phi1, phi2, loss=None):
        """
        Choose which phi the patient prefers. To match Fauvel 2021, the
        return value is 1 if the patient prefers phi1, and the return
        value is 0 if the patient prefers phi2.

        Parameters:
        -----------
        target : array
            The target image to use for the duel
        phi1, phi2 : 1D array
            Two sets of patient specific parameters
        loss : str or callable
            The loss function to use. If not provided, defaults to self.loss_type

        Returns:
        --------
        decision, logdict
        """

        def fix_shape(tensor):
            if len(tensor.shape) == 2:
                return tensor[None, ..., None]
            elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
                return tensor[..., None]
            elif len(tensor.shape) == 3 and tensor.shape[-1] == 1:
                return tensor[None, ...]

        target = fix_shape(target)
        if loss is None:
            loss = self.loss_type

        stim1, percept1, logs1 = self.inv_target(target, phi1)
        stim2, percept2, logs2 = self.inv_target(target, phi2)

        percept1 = fix_shape(percept1)
        percept2 = fix_shape(percept2)

        loss1 = self.loss(target, percept1, fn=loss)
        loss2 = self.loss(target, percept2, fn=loss)

        decision = self.simulate_decision(loss1, loss2)       

        def format_dict(l):
            return {
                'percept' : l[0],
                'loss' : l[1],
                'stim': l[2],
                'logs' : l[3]
            }

        ret_dict = {
            'phi1' : format_dict([percept1, loss1, stim1, logs1]),
            'phi2' : format_dict([percept2, loss2, stim2, logs2]),
            'target' : target,
            'decision' : decision
        }

        return decision, ret_dict
    
    def duel_plot(self, ret_dict):
        ncols = 3
        if 'naive' in ret_dict.keys() and ret_dict['naive']['percept'] is not None:
            ncols += 1
        
        fig, axes = plt.subplots(1, ncols)

        plt.sca(axes[0])
        plt.imshow(np.array(ret_dict['target']).squeeze(), cmap='gray')
        plt.title('Target')
        xlabel = f"choice:{ret_dict['decision']}"
        if 'extra_loss' in ret_dict.keys():
            xlabel += f"\n({ret_dict['extra_loss']['decision']:.2f}, {ret_dict['extra_loss']['phi1']:.2f}, {ret_dict['extra_loss']['phi2']:.2f})"
        plt.xlabel(xlabel)

        plt.sca(axes[1])
        plt.imshow(np.array(ret_dict['phi1']['percept']).squeeze(), cmap='gray')
        plt.title(fr"(1) $\phi_1$: {ret_dict['phi1']['loss']:.3f}")
        plt.xlabel(f"{ret_dict['phi1']['logs']['time']*1000:.0f}ms")

        plt.sca(axes[2])
        plt.imshow(np.array(ret_dict['phi2']['percept']).squeeze(), cmap='gray')
        plt.title(fr"(0) $\phi_2$: {ret_dict['phi2']['loss']:.3f}")
        plt.xlabel(f"{ret_dict['phi2']['logs']['time']*1000:.0f}ms")

        if ncols > 3:
            plt.sca(axes[3])
            plt.imshow(np.array(ret_dict['naive']['percept']).squeeze(), cmap='gray')
            plt.title(f"n: {ret_dict['naive']['loss']:.3f}")


        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()

        return fig, axes