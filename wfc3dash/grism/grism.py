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

# os.chdir(HOME_PATH)

import astropy.wcs as pywcs
import astropy.io.fits as pyfits
import astropy.units as u

# https://github.com/gbrammer/grizli
import grizli
from grizli import utils, model, multifit, prep
from grizli.pipeline import auto_script
from grizli.aws import db, visit_processor

# https://github.com/gbrammer/golfir
import golfir.utils

PATHS = None

def process_association(assoc='j100028p0215_0619_ehn_cosmos-g141-101_wfc3ir_g141', clean=True, align_kwargs={}, sync=True, **kwargs):
    """
    Process a DASH grism association
    """
    from stwcs import updatewcs    
    from wfc3dash import process_raw
    import astropy.time
    import astropy.table
    
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
    if __name__ == '__main__':
        import wfc3dash.grism.grism
        wfc3dash.grism.grism.PATHS = PATHS
        
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
        if not os.path.exists(file):
            prep.fresh_flt_file(file)
            updatewcs.updatewcs(os.path.basename(file), verbose=False, 
                            use_db=False)
    
    # By "visit", which are the DASH groups
    os.chdir(PATHS['prep'])
    visits = []
    qfiles = glob.glob('../RAW/*q_flt.fits')
    qfiles.sort()
    
    for f in qfiles:
        product = os.path.basename(f).split('q_flt.fits')[0]
        visit = {'files':glob.glob(f'{product}[a-o]_flt.fits'), 
                 'product':product}
        visit['files'].sort()
        visits.append(visit)
    
    failed = []
    
    for visit in visits:
        if not os.path.exists(f"{visit['product']}_column.png"):
            prep.visit_grism_sky(grism=visit)

        if not os.path.exists(visit['product']+'.log.txt'):
            try:
                align_visit(visit, **align_kwargs)
            except:
                failed.append(visit['product'])
                msg = f"{visit['product']} failed"
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                
                os.system(f"rm {visit['product']}*")
                continue
                
        footprints = []
        for file in visit['files']:
            im = pyfits.open(file)
            wcs = pywcs.WCS(im[1].header, relax=True)
            footprints.append(utils.SRegion(wcs).shapely[0])
        
        visit['footprints'] = footprints
    
    ok_visits = []
    for v in visits:
        if v['product'] not in failed:
            ok_visits.append(v)
            
    visits = ok_visits
    
    #### Visit log
    files = []
    for v in visits:
        #files.extend([os.path.join('./Prep/', f) for f in v['files']])
        files.extend([f for f in v['files']])
    
    info = utils.get_flt_info(files)
    
    auto_script.write_visit_info(visits, [], info, root=assoc, path='./')
    
    #### First sync
    visit_processor.update_assoc_status(assoc, status=2)
    
    #### Add alignment tables
    files = glob.glob('*log.txt')
    files.sort()
    
    if len(files) == 0:
        visit_processor.update_assoc_status(assoc, status=9)
        return False
        
    logs = []
    for file in files:
        with open(file) as fp:
            lt = utils.read_catalog(''.join(fp.readlines()[1:]))
            logs.append(lt)
    
    lt = astropy.table.vstack(logs)
    lt['assoc'] = assoc
    lt['modtime'] = astropy.time.Time.now().mjd
    
    ## to database
    db.execute(f"DELETE FROM dash_grism_alignment WHERE assoc = '{assoc}'")
    
    db.send_to_database('dash_grism_alignment', lt, if_exists='append')
    
    for visit in visits:
        visit_processor.exposure_info_from_visit(visit, assoc=assoc)
    
    ## files to s3
    os.system(f'aws s3 sync ./ s3://grizli-v2/HST/Pipeline/{assoc}/Prep/' + 
              ' --exclude "*" --include "iehn*png"' + 
              ' --include "iehn*txt" --include "*yaml"')
    
    #### Contamination model, etc.    
    compute_grism_contamination(assoc, **kwargs)
    
    #### Sync results
    if sync:
        sync_products(assoc)
        
    visit_processor.update_assoc_status(assoc, status=3)
    
    if clean:
        os.chdir(HOME_PATH)
        os.system(f'rm -rf {assoc}*')        


def sync_products(assoc):
    """
    Sync results to S3
    """
    os.chdir(os.path.join(HOME_PATH, assoc))
        
    os.system(f'rm Prep/*bkg.fits')
    
    drz_files = glob.glob('Prep/*_dr*fits')
    drz_files += glob.glob('Prep/*seg.fits')
    drz_files += glob.glob('Extractions/*_grism*[ni].fits')
    drz_files.sort()
        
    for file in drz_files:
        cmd = f'gzip {file}'
        print(cmd)
        os.system(cmd)
        
    os.system(f"""aws s3 sync ./ s3://grizli-v2/HST/Pipeline/{assoc}/ \
                              --exclude "*" \
                              --include "Prep/*flt.fits" \
                              --include "Prep/*txt" \
                              --include "Extractions/*" \
                              --include "Prep/{assoc}*" \
                              --include "Prep/iehn*[nx][gt]"
                              """)


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


def compute_grism_contamination(assoc, filters=['F814W','F105W','F140W','F125W','F160W'], grism_files=None, phot_kwargs={}, direct_filt='ir', 
tile_mosaics=True, refine_mag_limits=[18,23], **kwargs):
    """
    Extra steps for grism processing:
    
    1) Imaging mosaic, catalog, segmentation image
    
    2) Contamination model
    
    3) Drizzled model images
    
    """
    import golfir.catalog
    from . import tiles as mosaic_tiles
    
    global PATHS
    
    root = assoc    
    
    os.chdir(os.path.join(HOME_PATH, assoc, 'Prep'))
    
    if grism_files is None:
        files = glob.glob('*flt.fits')
        files.sort()
    
        # Remove q files from DASH-grism
        for j in range(len(files))[::-1]:
            _file = files[j]
            if _file.startswith('iehn') & ('q_flt' in _file):
                _ = files.pop(j)
            
        info = utils.get_flt_info(files)
        is_grism = np.array([f.startswith('G') for f in info['FILTER']])
        grism_files = info['FILE'][is_grism].tolist()
    
    if not os.path.exists(f'{root}-{direct_filt}_drz_sci.fits'):
        if tile_mosaics:
            #### Build mosaic from pre-drizzled tiles and catalog
            mosaic_tiles.create_mosaic_from_tiles(assoc, filt=direct_filt)
        else:
            ### Drizzle reference mosaic from imaging exposures
            _h, ir_wcs = utils.make_maximal_wcs(grism_files, pixel_scale=0.1, 
                                                get_hdu=False, pad=128)
                                        
            #### Reference mosaic
            visit_processor.cutout_mosaic(rootname=root, 
                                          skip_existing=False,
                                          ir_wcs=ir_wcs,
                                          filters=filters, 
                                          kernel='square', s3output=None, 
                                          gzip_output=False, clean_flt=True)
            
            #### Combined detection image
            golfir.catalog.make_charge_detection(root, ext='ir')
            
            #### Photometric catalog
            auto_script.multiband_catalog(field_root=root, **phot_kwargs)
    
    grp = auto_script.grism_prep(field_root=root, 
                                 gris_ref_filters={'G141':[direct_filt]},
                                 files=grism_files,
                                 refine_mag_limits=refine_mag_limits)
    
    #### Drizzled grp objects
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
    
    #### Fit params
    pline = auto_script.DITHERED_PLINE.copy()
    args_file = f'{root}_fit_args.npy'

    if (not os.path.exists(args_file)):
        
        msg = '# generate_fit_params: ' + args_file
        utils.log_comment(utils.LOGFILE, msg, verbose=True, show_date=True)

        pline['pixscale'] = 0.1 #mosaic_args['wcs_params']['pixel_scale']
        pline['pixfrac'] = 0.5  #mosaic_args['mosaic_pixfrac']
        if pline['pixfrac'] > 0:
            pline['kernel'] = 'square'
        else:
            pline['kernel'] = 'point'

        min_sens = 1.e-4
        min_mask = 1.e-4
        
        fit_trace_shift = True
        
        auto_script.generate_fit_params(field_root=root, prior=None, MW_EBV=0.0, pline=pline, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, min_sens=min_sens, min_mask=min_mask, sys_err=0.03, fcontam=0.2, zr=[0.05, 3.4], save_file=args_file, fit_trace_shift=fit_trace_shift, include_photometry=False, use_phot_obj=False)
        
        os.system(f'cp {args_file} fit_args.npy')


def fit_beam_shift(mb, xbounds=((-5,5), (-4,4)), ybounds=((-5,5), (-5,5)), verbose=True):
    """
    """
    from scipy.optimize import minimize
    from wfc3dash.grism.alignment import add_ytrace_offset
    
    spl = utils.bspline_templates(wave=np.arange(8000, 1.8e4, 10), df=3)
    
    tfit = mb.template_at_z(templates=spl)
    #fig = mb.oned_figure(tfit=tfit, units='flam') #, trace_limits=[-2,2])

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
        self.xoffset = shift[0]
        add_ytrace_offset(self, shift[1])

        tfit = mb.template_at_z(templates=spl)
        if verbose > 1:
            print(shift, tfit['chi2'])

        return tfit['chi2'] + np.sum(np.array(shift)**2/2/1)*2
    
    chi0 = objfun_sens([0,0])
    
    # First fit
    _x = minimize(objfun_sens, np.zeros(2), bounds=(xbounds[0], ybounds[0]), 
                  method='powell')
    
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
    _x = minimize(objfun_sens, np.zeros(2), bounds=(xbounds[1], ybounds[1]), 
                  method='powell')
    chi_final = objfun_sens(_x.x)
    
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
    #chi_final = tfit['chi2'] #+ np.sum(_x.x**2/2/1)*2

    shift = _x.x

    if (chi_final - chi0 < 2) & (np.abs(shift).max() < 4.8):
        #fig = mb.oned_figure(tfit=tfit, units='flam') #, trace_limits=[-2,2])
        print(' apply shift:', shift, chi_final - chi0)
    else:
        print('  -    shift:', shift, chi_final - chi0)
        #fig = mb.oned_figure(tfit=tfit, units='eps') #, trace_limits=[-2,2])


def extract_spectrum(grp, id, size=32):
    """
    """
    args = np.load('fit_args.npy', allow_pickle=True)[0]
    
    args['min_sens'] = 1.e-4
    args['min_mask'] = 1.e-4
        
    beams = grp.get_beams(id, size=size, min_mask=args['min_mask'], 
                          min_sens=args['min_sens'], mask_resid=False)
                              
    mb = run_align_dash(beams, args)
    return mb


def run_align_dash(beams, args):
    
    spl = utils.bspline_templates(wave=np.arange(8000, 1.8e4, 10), df=3)
    
    mb = multifit.MultiBeam(beams, group_name=args['group_name'], 
                            fcontam=args['fcontam'], 
                            min_sens=1.e-4,
                            min_mask=1.e-4, psf=False)
    
    tfit = mb.template_at_z(templates=spl)
    fig = mb.oned_figure(tfit=tfit, units='flam')
    fig.savefig(f"{args['group_name']}_{beams[0].id:05d}.init.1d.png")
    hdu, fig = mb.drizzle_grisms_and_PAs(tfit=tfit, diff=True)
    fig.savefig(f"{args['group_name']}_{beams[0].id:05d}.init.2d.png")
    
    for i in range(mb.N):
        mbi = multifit.MultiBeam(mb.beams[i:i+1], fcontam=0.,
                                 min_sens=0.0001, min_mask=0.0001)
        
        #tr = mbi.fit_trace_shift(tol=1.e-3)
        mbi.initialize_masked_arrays()
        
        fit_beam_shift(mbi)
        #print(i)
        
    #
    for b in mb.beams:
        b._parse_from_data(**b._parse_params)
        fm = nd.binary_dilation(b.fit_mask.reshape(b.sh), iterations=5)
        b.fit_mask = fm.flatten() & (b.ivarf > 0)

    mb.compute_model()

    mb._parse_beam_arrays()
    mb.initialize_masked_arrays()
    mb.write_master_fits()
    
    tfit = mb.template_at_z(templates=spl)
    fig = mb.oned_figure(tfit=tfit, units='flam')
    fig.savefig(f"{args['group_name']}_{beams[0].id:05d}.offs.1d.png")
    hdu, fig = mb.drizzle_grisms_and_PAs(tfit=tfit, diff=True)
    fig.savefig(f"{args['group_name']}_{beams[0].id:05d}.offs.2d.png")
    
    plt.close('all')
    
    return mb
    

def get_random_visit():
    """
    Find a visit that needs processing
    """

    all_assocs = db.SQL(f"""SELECT DISTINCT(assoc_name) 
                           FROM assoc_table
                           WHERE proposal_id = '16443'
                           AND status=12""")
    
    if len(all_assocs) == 0:
        return None
    
    random_assoc = all_assocs[np.random.randint(0, len(all_assocs))][0]
    return random_assoc


def run_one(clean=2, sync=True, align_kwargs={}):
    """
    Run a single random visit
    """
    import os
    import time

    nassoc = db.SQL("""SELECT count(distinct(assoc_table))
                       FROM assoc_table
                       WHERE proposal_id = '16443'
                       AND status = 12""")['count'][0] 
    
    assoc = get_random_visit()
    if assoc is None:
        with open('/GrizliImaging/dash_finished.txt','w') as fp:
            fp.write(time.ctime() + '\n')
    else:
        print(f'============  Run association  ==============')
        print(f'{assoc}')
        print(f'========= {time.ctime()} ==========')
        
        with open('/GrizliImaging/dash_visit_history.txt','a') as fp:
            fp.write(f'{time.ctime()} {assoc}\n')
        
        #process_visit(assoc, clean=clean, sync=sync)
        process_association(assoc=assoc, clean=clean,
                            align_kwargs=align_kwargs,
                            sync=sync)
                            