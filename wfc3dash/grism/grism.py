"""
Process DASH grism files from 
"""

import os
import glob

import numpy as np
import scipy.ndimage as nd

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.interpolation'] = 'Nearest'

#HOME_PATH = '/home/ec2-user/telescopes/notebooks/Dash'
HOME_PATH = '/GrizliImaging/'

os.chdir(HOME_PATH)

import astropy.wcs as pywcs
import astropy.io.fits as pyfits
import astropy.units as u

import grizli
from grizli import utils, model, multifit, prep
from grizli.pipeline import auto_script
from grizli.aws import db, visit_processor

import golfir.utils

PATHS = None

def process_association(assoc='j100028p0215_0619_ehn_cosmos-g141-101_wfc3ir_g141', clean=True, align_kwargs={}, **kwargs):
    """
    Process a DASH grism association
    """
    from stwcs import updatewcs    
    from wfc3dash import process_raw
    
    if __name__ == '__main__':
        from wfc3dash.grism.grism import align_grism
    
    global PATHS
    
    os.chdir(HOME_PATH)
    
    os.environ['orig_iref'] = os.environ.get('iref')
    os.environ['orig_jref'] = os.environ.get('jref')
    visit_processor.set_private_iref(assoc)
    
    visit_processor.update_assoc_status(assoc, status=1)
    
    tab = db.SQL(f"""SELECT * FROM assoc_table
                     WHERE assoc_name='{assoc}'""")
    
    if len(tab) == 0:
        print(f"assoc_name='{assoc}' not found in assoc_table")
        return False
        
    keep = []
    for c in tab.colnames:
        if tab[c].dtype not in [object]:
            keep.append(c)
    
    tab[keep].write(f'{assoc}_footprint.fits', overwrite=True)
    
    PATHS = auto_script.create_path_dict(assoc)

    auto_script.fetch_files(assoc, s3_sync=True, reprocess_clean_darks=False, 
                            reprocess_parallel=False)
    
    # DASHify the exposures
    os.chdir(PATHS['raw'])
    process_raw.run_all()
    
    # FLT copies
    os.chdir(PATHS['prep'])
    
    files = glob.glob(f'../RAW/*[a-o]_flt.fits')
    files.sort()
    
    for rfile in files:
        file = os.path.basename(rfile)
        prep.fresh_flt_file(file)
        updatewcs.updatewcs(os.path.basename(file), verbose=False, 
                            use_db=False)
    
    # By "visit", which are the DASH groups
    visits = []
    qfiles = glob.glob('../RAW/*q_flt.fits')
    qfiles.sort()
    
    for f in qfiles:
        product = os.path.basename(f).split('q_flt.fits')[0]
        visit = {'files':glob.glob(f'{product}[a-o]_flt.fits'), 
                 'product':product}
        visit['files'].sort()
        visits.append(visit)
    
    for visit in visits:
        if not os.path.exists(f"{visit['product']}_column.png"):
            prep.visit_grism_sky(grism=visit)

        align_visit(visit, **align_kwargs)
    
        footprints = []
        for file in visit['files']:
            im = pyfits.open(file)
            wcs = pywcs.WCS(im[0].header, relax=True)
            footprints.append(utils.SRegion(wcs).shapely[0])
        
        visit['footprints'] = footprints
        
    finish_group(assoc, **kwargs)


def align_visit(visit, flag_crs=True, driz_cr_kwargs={'driz_cr_snr_grow':3}, **kwargs):
    """
    """
    from skimage.transform import SimilarityTransform
    
    from tristars import match
    from .alignment import align_dash_exposure

    global PATHS
    

    os.chdir(PATHS['prep'])
    ref_file = os.path.join(HOME_PATH, 
                            'uvista_DR4_Ks_ADP.2019-02-05T17_00_54.618.fits')
    eso_file = 'https://dataportal.eso.org/dataPortal/file/'
    eso_file += 'ADP.2019-02-05T17:00:54.618'
    if not os.path.exists(ref_file):
        os.system(f'wget {eso_file} -O {ref_file}')

    kcat = utils.read_catalog(ref_file)
    kcat['H_AUTO'] = kcat['MAG_AUTO']
    
    tabs = {}
    for file in visit['files']:
        print(f'============\n{file}\n============')
        if file in tabs:
            continue
        tabs[file] = align_dash_exposure(flt_file=file, verbose=0)
        
    plt.close("all")
    
    LOGFILE = visit['product']+'.log.txt'

    # WCS alignment
    msg = f'# file rmsx rmsy dx_arcsec dy_arcsec'
    utils.log_comment(LOGFILE, msg, verbose=True, show_date=True)
    
    for k in tabs:
        C1 = np.array([tabs[k]['ra'], tabs[k]['dec']]).T

        idx, dr = tabs[k].match_to_catalog_sky(kcat)

        mlim = 20.2 if len(kcat) > 2000 else 21.5

        try:
            clip = (dr < 2*u.arcmin) & (kcat['H_AUTO'] < mlim)
        except:
            clip = (dr < 1*u.arcmin) & (kcat['MAG_AUTO'] < 20.2)

        C2 = np.array([kcat['ALPHA_J2000'][clip], 
                       kcat['DELTA_J2000'][clip]]).T

        x0 = np.mean(C1, axis=0)
        V1 = (C1-x0)*3600
        V2 = (C2-x0)*3600

        pair_ix = match.match_catalog_tri(V1, V2, ignore_rot=True, 
                                          ignore_scale=True,
                                          size_limit=[1,110], 
                                          auto_keep=False)

        tfo, dx, rms = match.get_transform(V1, V2, pair_ix,
                                           transform=SimilarityTransform, 
                                           use_ransac=True)

        ok = np.abs(dx.max(axis=1)) < 1

        tfo, dx, rms = match.get_transform(V1, V2, pair_ix[ok,:], 
                                           transform=SimilarityTransform, 
                                           use_ransac=True)

        fig = match.match_diagnostic_plot(V1, V2, pair_ix[ok,:], tf=tfo,
                                                  new_figure=True)
        
        fig.axes[0].set_xlim(-180, 180)
        fig.axes[0].set_ylim(-180, 180)
        
        fig.savefig(k.split('_flt')[0] + '.wcs.png')
        
        #print(k, rms, tfo.translation)
        tr = tfo.translation
        
        msg = f'{k} {rms[0]:.2f} {rms[1]:.2f} {tr[0]:.2f} {tr[1]:.2f}'
        utils.log_comment(LOGFILE, msg, verbose=True, show_date=False)
        
        im = pyfits.open(k, mode='update')
        wcs = pywcs.WCS(im[1].header)
        cosd = np.array([np.cos(wcs.wcs.crval[1]/180*np.pi), 1])
        ddeg = tfo.translation/3600 * cosd
        wcs.wcs.crval += ddeg

        h = utils.to_header(wcs)
        for hk in h:
            if hk in im[1].header:
                im[1].header[hk] = h[hk]
        
        if 'WCS_DX' in im[0].header:
            im[0].header['WCS_DX'] += tfo.translation[0]
            im[0].header['WCS_DY'] += tfo.translation[1]
        else:
            im[0].header['WCS_DX'] = tfo.translation[0], 'X WCS shift, arcsec'
            im[0].header['WCS_DY'] = tfo.translation[1], 'Y WCS shift, arcsec'
            
        im.flush() #writeto('x'+im.filename(), overwrite=True)
    
    plt.close('all')
    
    if flag_crs:
        prep.drizzle_overlaps([visit], parse_visits=False, 
                              check_overlaps=False, scale=0.150, 
                              pixfrac=1, final_kernel='point',
                              run_driz_cr=True, skysub=False, 
                              reset_bits=4096,
                              **driz_cr_kwargs)


def finish_group(assoc, filters=['F814W','F105W','F140W','F125W','F160W'], phot_kwargs={}, **kwargs):
    """
    Things to do as a group
    """
    
    global PATHS
    
    os.chdir(os.path.join(HOME_PATH, assoc, 'Prep'))
    
    grism_files = glob.glob('iehn*flt.fits')
    grism_files.sort()
    
    _h, ir_wcs = utils.make_maximal_wcs(grism_files, pixel_scale=0.1, 
                                        get_hdu=False, pad=64)
                                        
    #### Reference mosaic
    root = assoc    
    visit_processor.cutout_mosaic(rootname=root, 
                              skip_existing=False,
                              ir_wcs=ir_wcs,
                              filters=filters, 
                              kernel='square', s3output=None, 
                              gzip_output=False, clean_flt=True)
    
    golfir.catalog.make_charge_detection(root, ext='ir')
    
    auto_script.multiband_catalog(field_root=root, **phot_kwargs)
    
    grp = auto_script.grism_prep(field_root=root, 
                                 gris_ref_filters={'G141':['ir']},
                                 files=grism_files,
                                 refine_mag_limits=[18,23])
    
    if len(glob.glob(f'{root}*_grism*fits*')) == 0:
        grism_files = glob.glob('*GrismFLT.fits')
        grism_files.sort()

    # Drizzled grp objects
    # All files
    if len(glob.glob(f'{root}*_grism*fits*')) == 0:
        grism_files = glob.glob('*GrismFLT.fits')
        grism_files.sort()

        catalog = glob.glob(f'{root}-*.cat.fits')[0]
        try:
            seg_file = glob.glob(f'{root}-*_seg.fits')[0]
        except:
            seg_file = None

        grp = multifit.GroupFLT(grism_files=grism_files, direct_files=[], 
                                ref_file=None, seg_file=seg_file, 
                                catalog=catalog, cpu_count=-1, sci_extn=1, 
                                pad=256)

        # Make drizzle model images
        grp.drizzle_grism_models(root=root, kernel='point', scale=0.15)

        # Free grp object
        del(grp)
    
    