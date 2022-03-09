import numpy as np
from skimage.feature import peak_local_max
from skimage.feature import match_template

import astropy.io.fits as pyfits

import scipy.ndimage as nd
from scipy.optimize import minimize

from grizli import multifit, prep, utils, model
import golfir.utils

spl = utils.bspline_templates(wave=np.arange(8000, 1.8e4, 10), df=3)

def add_ytrace_offset(self, yoffset):
    """Add an offset in Y to the spectral trace

    Parameters
    ----------
    yoffset : float
        Y-offset to apply

    """
    from grizli.utils_c import interp
    
    self.ytrace_beam, self.lam_beam = self.conf.get_beam_trace(
                        x=(self.xc+self.xcenter-self.pad)/self.grow,
                        y=(self.yc+self.ycenter-self.pad)/self.grow,
                        dx=(self.dx+self.xcenter*0+self.xoff)/self.grow,
                        beam=self.beam, fwcpos=self.fwcpos)

    self.ytrace_beam *= self.grow
    self.yoffset = yoffset

    self.ytrace_beam += yoffset

    # Integer trace
    # Add/subtract 20 for handling int of small negative numbers
    dyc = np.cast[int](self.ytrace_beam+20)-20+1

    # Account for pixel centering of the trace
    self.yfrac_beam = self.ytrace_beam - np.floor(self.ytrace_beam)

    try:
        self.flat_index = self.idx[dyc + self.x0[0], self.dxpix]
    except IndexError:
        # print 'Index Error', id, self.x0[0], self.xc, self.yc, self.beam, self.ytrace_beam.max(), self.ytrace_beam.min()
        raise IndexError

    # Trace, wavelength, sensitivity across entire 2D array
    self.ytrace, self.lam = self.conf.get_beam_trace(
                        x=(self.xc+self.xcenter-self.pad)/self.grow,
                        y=(self.yc+self.ycenter-self.pad)/self.grow,
                        dx=(self.dxfull+self.xcenter+self.xoff)/self.grow,
                        beam=self.beam, fwcpos=self.fwcpos)

    self.ytrace *= self.grow
    self.ytrace += yoffset

    # Reset sensitivity
    ysens = self.lam_beam*0
    so = np.argsort(self.lam_beam)

    conf_sens = self.conf.sens[self.beam]
    if self.MW_F99 is not None:
        MWext = 10**(-0.4*(self.MW_F99(conf_sens['WAVELENGTH']*u.AA)))
    else:
        MWext = 1.

    ysens[so] = interp.interp_conserve_c(self.lam_beam[so],
                                         conf_sens['WAVELENGTH'],
                                         conf_sens['SENSITIVITY']*MWext,
                                         integrate=1, left=0, right=0)
    self.lam_sort = so
    self.sensitivity_beam = ysens

    # Full array
    ysens = self.lam*0
    so = np.argsort(self.lam)
    ysens[so] = interp.interp_conserve_c(self.lam[so],
                                         conf_sens['WAVELENGTH'],
                                         conf_sens['SENSITIVITY']*MWext,
                                         integrate=1, left=0, right=0)

    self.sensitivity = ysens


#
def get_beam_kernel(file):
    """
    Kernel for source identification
    """
    from grizli import model
    
    # Kernel for matching
    flt = model.GrismFLT(grism_file=file, sci_extn=1, pad=0)
    size = 16
    cut = flt.compute_model_orders(x=507, y=507, size=16, get_beams='A', 
                                   in_place=False) #, psf_params=[0,0])
    
    beam = model.BeamCutout(flt=flt, beam=cut['A'], conf=flt.conf)
    beam.init_epsf(psf_params=[0,0], get_extended=True)
    
    beam.psf_params = np.array([0,0])

    xspec = np.arange(5000,2.e4)
    yspec = (xspec/1.3e4)**-1.

    beam.compute_model(use_psf=True, spectrum_1d=(xspec, yspec*1.e-19), is_cgs=True)
    kernel = beam.model*1.#/beam.model.sum()

    for r in [-2,-1,1,2]:
        kernel += np.roll(beam.model, r, axis=0)*2.**-np.abs(r)

    kernel /= kernel.sum()

    k0 = kernel*1
    kx = kernel*1
    kx[:, np.abs(beam.wave-1.39e4) < 2300] = 0

    #plt.imshow(kernel)

    beam.sh

    conf = beam.beam.conf    
    return beam, conf, kernel, kx, k0


def align_dash_exposure(flt_file='iehn5vr8a_flt.fits', verbose=0):
    """
    """
    import matplotlib.pyplot as plt
    
    # Detect grism "objects"
    im = pyfits.open(flt_file)
    
    utils.unset_dq_bits(im['DQ'].data, 512)

    sci = im['SCI'].data*1
    dq = im['DQ'].data > 0
    var = im['ERR'].data**2
    ivar = 1/var
    #ivar = ivar*0.+np.median(ivar[~dq])
    dq |= ~np.isfinite(sci+ivar)

    # quick CRs
    fthresh=0.2
    kern = np.zeros((101,21))
    kern[50,10] = 1
    filt = match_template(sci, kern.T, pad_input=True)

    sat = filt > fthresh
    nsat = 4

    for g in range(nsat):
        sat = nd.binary_dilation(sat) & (sci > 7*im['ERR'].data)

    dq |= sat

    ivar[dq] = 0

    sky = np.median(sci[~dq])
    
    beam, conf, kernel, kx, k0 = get_beam_kernel(flt_file)
    
    cnum = golfir.utils.convolve_helper((sci-sky)*ivar, kernel)
    cden = golfir.utils.convolve_helper(ivar, kernel**2)
    cden2 = golfir.utils.convolve_helper(ivar, kernel)

    cnumx = golfir.utils.convolve_helper((sci-sky)*ivar, kx)
    cdenx = golfir.utils.convolve_helper(ivar, kx**2)

    flt_sn = cnum*np.sqrt(cden)
    nb_labels = 0
    threshold = 100

    while nb_labels < 16:
        threshold *= 0.9
        label_objects, nb_labels = nd.label(flt_sn > threshold)

    #print(threshold, nb_labels)

    fp = k0 > 0.001*k0.max()

    yc, xc = peak_local_max(flt_sn, footprint=fp*1, threshold_abs=threshold, 
                            labels=label_objects, 
                            num_peaks_per_label=1).T

    clip = (xc > 100) & (xc < 1014-100)
    xc = xc[clip]
    yc = yc[clip]
    
    # Make figure
    src_fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(label_objects, cmap='rainbow')
    ax.scatter(xc, yc, color='r', marker='x')
    src_fig.savefig(flt_file.split('_flt')[0]+'.src.png')
    
    # project grism positions based on trace

    #conf = beam.beam.conf
    x0 = 103
    dy0, lam0 = conf.get_beam_trace(xc, yc, x0)
    dy1, lam1 = conf.get_beam_trace(xc, yc, x0+1)

    dy, lam_ref = conf.get_beam_trace(507, 507, x0)

    dlam = lam1-lam0
    dx = -(lam0-lam_ref)/dlam + x0

    import astropy.wcs 
    wcs = astropy.wcs.WCS(im['SCI'].header, relax=True)
    xd, yd = xc-dx, yc-dy0-dy
    
    has_shift = xd > 1e10
    
    #####################
    # Refine positions
    pad = 100
    flt = model.GrismFLT(grism_file=flt_file, sci_extn=1, pad=pad)
    
    for i in range(len(xd)):
        size = 16

        print(i, xd[i]+pad, yd[i]+pad)
        #ds9.set(f'pan to {xd[i]+pad+1} {yd[i]+pad+1}')

        cut = flt.compute_model_orders(x=xd[i]+pad, y=yd[i]+pad, size=size,
                                       get_beams='A', in_place=False)
        beam = model.BeamCutout(flt=flt, beam=cut['A'], conf=flt.conf, 
                                min_sens=0.001, min_mask=0.001)

        # centered PSF as reference
        beam.beam.total_flux = 1
        beam.init_epsf(psf_params=[0,0], get_extended=True)
        psf = np.abs(beam.beam.psf)*1

        beam = model.BeamCutout(flt=flt, beam=cut['A'], conf=flt.conf, min_sens=0.001, min_mask=0.001)
        beam.beam.direct = psf.astype(np.float32)*1
        beam.beam.total_flux = beam.beam.direct.sum()
        beam.beam.seg = (psf > 0.01*psf.max()).astype(np.float32)*(i+1)

        beam.id = beam.beam.id = int(beam.beam.seg.max())
        beam.beam.set_segmentation(beam.beam.seg)

        beam._parse_from_data(**beam._parse_params)
        beam.compute_model()
        
        ###### Multibeam
        mb = multifit.MultiBeam([beam], root='test', min_sens=-1, min_mask=-1, mask_resid=False, fcontam=0)

        # Initial chi-squared
        tfit = mb.template_at_z(templates=spl)
        chi0 = tfit['chi2']

        # Grow fit mask
        for b in mb.beams:
            b._parse_from_data(**b._parse_params)
            fm = nd.binary_dilation(b.fit_mask.reshape(b.sh), iterations=5)
            b.fit_mask = fm.flatten() & (b.ivarf > 0)

        mb.compute_model()

        mb._parse_beam_arrays()
        mb.initialize_masked_arrays()

        def objfun_sens(shift):
            #print(shift)

            self = mb.beams[0].beam
            self.xoff = shift[0]
            add_ytrace_offset(self, shift[1])

            tfit = mb.template_at_z(templates=spl)
            if verbose > 1:
                print(shift, tfit['chi2'])
            
            return tfit['chi2']

        # First fit
        _x = minimize(objfun_sens, np.zeros(2), bounds=((-5,5), (-5, 5)), method='powell')
        
        # Reset fit mask
        for b in mb.beams:
            # b._parse_from_data()
            b._parse_from_data(**b._parse_params)

            # Needed for background modeling
            if hasattr(b, 'xp'):
                delattr(b, 'xp')

        mb._parse_beam_arrays()
        mb.initialize_masked_arrays()
        mb.compute_model()
        
        # Refit
        _x = minimize(objfun_sens, np.zeros(2), bounds=((-5,5), (-5, 5)), method='powell')
        
        # Reset fit mask
        for b in mb.beams:
            # b._parse_from_data()
            b._parse_from_data(**b._parse_params)

            # Needed for background modeling
            if hasattr(b, 'xp'):
                delattr(b, 'xp')

        mb._parse_beam_arrays()
        mb.initialize_masked_arrays()
        mb.compute_model()
        
        tfit = mb.template_at_z(templates=spl)
        chi_final = tfit['chi2']

        shift = _x.x

        if (chi_final - chi0 < -50) & (np.abs(shift).max() < 4.8):
            #fig = mb.oned_figure(tfit=tfit, units='eps') #, trace_limits=[-2,2])
            xd[i] += shift[0]
            yd[i] += shift[1]
            has_shift[i] = True
            print(' apply shift:',i, shift, chi_final - chi0)
        else:
            print('  -    shift:',i, shift, chi_final - chi0)
    
    tab = utils.GTable()
    tab['ra'], tab['dec'] = wcs.all_pix2world(xd, yd, 0)
    tab['xd'], tab['yd'] = xd, yd
    tab['has_shift'] = has_shift
    
    return tab

if 0:
    import glob
    import os
    os.chdir('/mnt/efs/fs1/telescopes/notebooks/Dash/j100232p0247/Prep')

    files = glob.glob('iehn5vrg[a-o]_flt.fits')

    os.chdir('/mnt/efs/fs1/telescopes/notebooks/Dash/j100232p0243/Prep')

    files = glob.glob('iehn5wb1[a-o]_flt.fits')
    # files = glob.glob('iehn5wba[a-o]_flt.fits')

    #files = glob.glob('iehn5vr8[a-o]_flt.fits')
    files.sort()
    tabs = {}
    for file in files:
        print(f'============\n{file}\n============')
        if file in tabs:
            continue
        tabs[file] = align_dash_exposure(flt_file=file, verbose=0)