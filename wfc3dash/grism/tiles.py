"""
Tiles in COSMOS
"""
import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imsave

from grizli import utils
from grizli.aws import db
from grizli.pipeline import auto_script


def make_cosmos_tiles():
    """
    Make tiles across COSMOS with a single tangent point
    """
    import numpy as np
    
    from grizli_aws import field_tiles
    from grizli.aws import tile_mosaic
    
    np.set_printoptions(precision=6)
    
    tile_npix = 4096
    pscale = 0.1
    
    ra, dec = 150.1, 2.29
    rsize = 48
    overlap = 0
    
    # New tiles with overlaps
    ra, dec = 150.125, 2.2
    rsize = 45
    pscale = 0.080
    tile_npix = 2048+256

    tile_arcmin = tile_npix*pscale/60
    overlap=tile_arcmin/9
    
    print('Tile size, arcmin = ', tile_arcmin)
    
    name='cos'
    tiles = field_tiles.define_tiles(ra=ra, dec=dec, 
                                           size=(rsize*2, rsize*2), 
                                           tile_size=tile_arcmin, 
                                           overlap=overlap, field=name, 
                                           pixscale=pscale, theta=0)
    
    filt = 'G141'
    extra = " AND file like 'iehn%%' "

    filt = 'F814W'
    extra = ''
    
    fig, tab = tile_mosaic.exposure_map(ra, dec, rsize, name, 
                                        filt=filt.upper(), s0=18, 
                                        extra=extra)
    fig.tight_layout(pad=0.5)
    fig.savefig('/tmp/map.png')
    fig.tight_layout(pad=0.5)
    fig.savefig('/tmp/map.png')
    
    ax = fig.axes[0]
    for t in tiles:
        wcs = tiles[t]
        sr = utils.SRegion(wcs.calc_footprint())
        ax.add_patch(sr.patch(fc='None', ec='r', alpha=0.5, 
                              transform=ax.get_transform('world'))[0])
        
        ax.text(*sr.centroid[0], t, ha='center', va='center', 
                transform=ax.get_transform('world'), fontsize=7)
    
    rows = []
    for t in tiles:
        wcs = tiles[t]
        sr = utils.SRegion(wcs.calc_footprint())
                    
        wcsh = utils.to_header(wcs)
        row = [t]
        for k in wcsh:
            row.append(wcsh[k])
        
        fp = sr.xy[0].__str__().replace('\n', ',').replace('   ', ',')
        fp = fp.replace(' ', '').replace('[','(')
        fp = fp.replace(']', ')').replace(',,',',')
        row.append(fp)     
        
        rows.append(row)
    
    names = ['tile']
    for k in wcsh:
        names.append(k.lower())
    
    names.append('footprint')
    
    cos_tile = utils.GTable(rows=rows, names=names)
    cos_tile['status'] = 0
    
    cos_tile['field'] = name
    
    #db.send_to_database('cosmos_tiles', cos_tile, if_exists='replace')
    db.send_to_database('combined_tiles', cos_tile, if_exists='replace')
    
    # To do: fix footprint in cosmos_tiles
    # ...
    
    # if 0:
    #     # Find tiles that overlap dash visit
    #     db.SQL("select distinct(tile) from cosmos_tiles_tmp t, exposure_files e where e.assoc = 'j100040p0216_2190_ehn_cosmos-g141-156_wfc3ir_g141' AND polygon(e.footprint) && polygon(t.footprint)")
    #     

def split_tiles(root='abell2744-080-08.08', ref_tile=(8,8), filters=['visr','f125w','h'], optical=False, suffix='.rgb', xsize=6, zoom_levels=[4,3,2,1], force=False, scl=1, invert=False, verbose=True, rgb_scl=[1,1,1], rgb_min=-0.01):
    """
    Split image into 256 pixel tiles for map display
    """
    nx = (2048+256)
    dpi = int(nx/xsize)
    
    if os.path.exists(f'{root}{suffix}.png') & (~force):
        return True
    
    try:
        _ = auto_script.field_rgb(root=root,
                                  xsize=xsize, filters=filters, 
                                  full_dimensions=2**optical, HOME_PATH=None, 
                                  add_labels=False,
                                  gzext='*', suffix=suffix, 
                                  output_format='png',
                                  scl=scl, invert=invert, 
                                  rgb_scl=rgb_scl, rgb_min=rgb_min)
    except IndexError:
        return False
    
    fig = _[-1]
    
    base = '_'.join(root.split('-')[:-1]).replace('+','_') + '_' + suffix[1:]
    
    tx, ty = np.cast[int](root.split('-')[-1].split('.'))

    for iz, zoom in enumerate(zoom_levels):
        if iz > 0:
            zoom_img = f'{root}{suffix}.{2**iz:d}.png'
            fig.savefig(zoom_img, dpi=dpi/2**iz)
            img = plt.imread(zoom_img)
        else:
            img = plt.imread(f'{root}{suffix}.png')
        
        if verbose:
            print(f'zoom: {zoom} {img.shape}')
        
        img = img[::-1,:,:]
        
        ntile = int(2048/2**(4-zoom)/256)
        left = (tx - ref_tile[0])*ntile
        bot = -(ty - ref_tile[1])*ntile+2*ntile
        # print(zoom, ntile, left, bot)

        #axes[iz].set_xlim(-ntile*0.1, ntile*(1.1)-1)
        #axes[iz].set_ylim(*axes[iz].get_xlim())

        for i in range(ntile):
            xi = left + i
            for j in range(ntile):
                yi = bot - j - 1
                
                slx = slice((i*256), (i+1)*256)
                sly = slice((j*256), (j+1)*256)
                
                tile_file = f'{root}-tiles/{base}/{zoom}/{yi}/{xi}.png'
                if verbose > 1:
                    print(f'  {i} {j} {tile_file}')
                
                dirs = tile_file.split('/')
                for di in range(1,5):
                    dpath = '/'.join(dirs[:di])
                    #print(dpath)
                    if not os.path.exists(dpath):
                        os.mkdir(dpath)
                                     
                imsave(tile_file, img[sly, slx, :][::-1,:,:],
                       plugin='pil', format_str='png')


def make_all_tile_images(root, force=False, ref_tile=(8,8), cleanup=True, zoom_levels=[4,3,2,1], brgb_filts=['visr','visb','uv'], rgb_filts=['visr','j','h'], blue_is_opt=True, make_opt_filters=True, make_ir_filters=True):
    
    #root = f'{field}-080-08.08'

    files = glob.glob(f'{root}-[hvu]*')
    files += glob.glob(f'{root}*.rgb.png')

    if len(files) == 0:
        auto_script.make_filter_combinations(root, 
                          filter_combinations={'h':['F140W','F160W'], 
                                               'j':['F105W','F110W','F125W'],
                                            'visr':['F850LP','F814W','F775W'],
                                     'visb':['F606W','F555W', 'F606WU'][:-1],
                                'uv':['F438WU','F435W','F435WU', 'F475W']}, 
                                            weight_fnu=False)

    if 'j' in rgb_filts:
        rgb_scl = [1.1, 0.8, 1]
    else:
        rgb_scl = [1,1,1]
        
    split_tiles(root, ref_tile=ref_tile, 
                filters=rgb_filts, zoom_levels=zoom_levels,
                optical=False, suffix='.rgb', xsize=6, scl=1,
                force=force, rgb_scl=rgb_scl)

    plt.close('all')
    
    if len(glob.glob(f'{root}*.brgb.png')) == 0:
        split_tiles(root, ref_tile=ref_tile, 
                    filters=brgb_filts, zoom_levels=zoom_levels,
                    optical=blue_is_opt, suffix='.brgb', xsize=6, scl=2,
                    force=force, rgb_scl=[1., 1.2, 1.4], rgb_min=-0.018)

        plt.close('all')
        
    if root.startswith('cos'):
        if len(glob.glob(f'{root}*.vi.png')) == 0:
            split_tiles(root, ref_tile=ref_tile, 
                        filters=['f814w','f160w'], zoom_levels=zoom_levels,
                        optical=False, suffix='.vi', xsize=6, scl=0.8,
                        force=force, rgb_scl=[1, 1, 1], rgb_min=-0.018)

            plt.close('all')
        
    # IR
    if make_ir_filters:
        files = glob.glob(f'{root}-f[01]*sci.fits*')
        files.sort()
        filts = [file.split(f'{root}-')[1].split('_')[0] for file in files]
        for filt in filts:
            if os.path.exists(f'{root}.{filt}.png'):
                continue

            split_tiles(root, ref_tile=ref_tile, 
                    filters=[filt], zoom_levels=zoom_levels,
                    optical=False, suffix=f'.{filt}', xsize=6, 
                    force=force, scl=2, invert=True)

            plt.close('all')
    
    if make_opt_filters:
        # Optical, 2X pix
        files = glob.glob(f'{root}-f[2-8]*sci.fits*')
        files.sort()
        filts = [file.split(f'{root}-')[1].split('_')[0] for file in files]
        for filt in filts:
            if os.path.exists(f'{root}.{filt}.png'):
                continue

            split_tiles(root, ref_tile=ref_tile, 
                filters=[filt], zoom_levels=zoom_levels,
                optical=True, suffix=f'.{filt}', xsize=6, 
                force=force, scl=2, invert=True)

            plt.close('all')
    
    if cleanup:
        files = glob.glob(f'{root}-[vhuj]*fits*')
        files.sort()
        for file in files:
            print(f'rm {file}')
            os.remove(file)


TILE_FILTERS = ['F336W', 'F350LP', 'F390W', 'F435W', 'F438W', 'F475W',
                'F555W', 'F606W', 'F625W', 'F775W', 
                'F814W', 'F850LP', 
                'F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W'][1:]

def process_tile(field='cos', tile='01.01', filters=TILE_FILTERS, fetch_existing=True, cleanup=True):
    """
    """
    import numpy as np
    
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from grizli.aws import visit_processor, db
    from grizli import utils
    import golfir.catalog
    from grizli.pipeline import auto_script
    
    utils.LOGFILE = 'mosaic.log'
    
    db.execute(f"update combined_tiles set status=1 where tile = '{tile}' and field='{field}'")
    
    row = db.SQL(f"select * from combined_tiles where tile = '{tile}' and field='{field}'")
    
    h = pyfits.Header()
    for k in row.colnames:
        if k in ['footprint']:
            continue
        
        h[k.upper()] = row[k][0]
        
    ir_wcs = pywcs.WCS(h)
    
    root = f'{field}-080-{tile}'
    
    if fetch_existing:
        os.system(f"""aws s3 sync s3://grizli-v2/ClusterTiles/{field}/ ./
                          --exclude "*"
                          --include "{root}*_dr?*fits.gz"
                          """.replace('\n', ' '))
        
        files = glob.glob(f'{root}*gz')
        for file in files:
            os.system(f'gunzip --force {file}')
            
    visit_processor.cutout_mosaic(rootname=root, 
                              skip_existing=True,
                              ir_wcs=ir_wcs,
                              filters=filters, 
                              kernel='square', s3output=None, 
                              gzip_output=False, clean_flt=True)
    
    files = glob.glob(f'{root}*dr*fits')
    if len(files) == 0:
        db.execute(f"update combined_tiles set status=4 where tile = '{tile}'")
        return True
        
    golfir.catalog.make_charge_detection(root, ext='ir')
    
    phot = auto_script.multiband_catalog(field_root=root) #, **phot_kwargs)
    
    if len(phot) == 0:
        # Empty catalog
        db.execute(f"update combined_tiles set status=10 where tile = '{tile}' AND field = '{field}'")
        
        if cleanup:

            files = glob.glob(f'{root}*')
            files.sort()
            for file in files:
                print(f'rm {file}')
                os.remove(file)
        
        return False
        
    for i in [4,5,6]:
        for c in phot.colnames:
            if c.endswith(f'{i}'):
                phot.remove_column(c)
    
    for c in phot.colnames:
        if c in ['ra','dec','x_world','y_world']:
            continue
            
        if phot[c].dtype == np.float64:
            phot[c] = phot[c].astype(np.float32)
        elif phot[c].dtype == np.int64:
            phot[c] = phot[c].astype(np.int32)
              
    phot['tile'] = tile
    phot['field'] = field
    
    if 'id' in phot.colnames:
        phot.remove_column('id')
    
    for c in ['xmin','xmax','ymin','ymax']:
        if c in phot.colnames:
            phot.rename_column(c, c+'pix')
    
    if 0:
        db.execute('CREATE TABLE combined_tile_phot AS SELECT * FROM cosmos_tile_phot limit 0')
        db.execute('ALTER TABLE combined_tile_phot DROP COLUMN id;')
        db.execute('ALTER TABLE combined_tile_phot ADD COLUMN id SERIAL PRIMARY KEY;')
        
    db.execute(f"DELETE from combined_tile_phot WHERE tile='{tile}' and field='{field}'")
    
    seg = pyfits.open(f'{root}-ir_seg.fits')
    
    ### IDs on edge
    edge = np.unique(seg[0].data[16:19,:])
    edge = np.append(edge, np.unique(seg[0].data[-19:-16,:]))
    edge = np.append(edge, np.unique(seg[0].data[:, 16:19]))
    edge = np.append(edge, np.unique(seg[0].data[:, -19:-16]))
    edge = np.unique(edge)
    phot['edge'] = np.in1d(phot['number'], edge)*1
    
    ### Add missing columns
    cols = db.SQL('select * from combined_tile_phot limit 2').colnames
    for c in phot.colnames:
        if c not in cols:
            print('Add column {0} to `combined_tile_phot` table'.format(c))
            if phot[c].dtype in [np.float64, np.float32]:
                SQL = "ALTER TABLE combined_tile_phot ADD COLUMN {0} real;".format(c)
            else:
                SQL = "ALTER TABLE combined_tile_phot ADD COLUMN {0} int;".format(c)
                
            db.execute(SQL)
            
    db.send_to_database('combined_tile_phot', phot, if_exists='append')
    
    if 'id' not in cols:
        # Add unique id index column
        db.execute('ALTER TABLE combined_tile_phot ADD COLUMN id SERIAL PRIMARY KEY;')
    
    # Use db id
    ids = db.SQL(f"SELECT number, id, ra, dec, tile from combined_tile_phot WHERE tile='{tile}' AND field='{field}'")
    idx, dr = ids.match_to_catalog_sky(phot)
        
    phot['id'] = phot['number']
    for c in ['xmin','xmax','ymin','ymax']:
        if c+'pix' in phot.colnames:
            phot.rename_column(c+'pix', c)
    
    golfir.catalog.switch_segments(seg[0].data, phot, ids['id'][idx])
    pyfits.writeto(f'{root}-ir_seg.fits', data=seg[0].data, 
                   header=seg[0].header, overwrite=True)
    
    ### Make subtiles
    ref_tiles = {'cos': (16,16)}
    
    if field in ref_tiles:
        ref_tile = ref_tiles[field]
    else:
        ref_tile = (9, 9)
        
    make_all_tile_images(root, force=False, ref_tile=ref_tile,
                         rgb_filts=['h','j','visr'],
                         brgb_filts=['visr','visb','uv'],
                         blue_is_opt=(field not in ['j013804m2156']), 
                         make_ir_filters=True, 
                         make_opt_filters=True)
    
    print(f'Sync ./{root}-tiles/ >> s3://grizli-v2/ClusterTiles/Map/{field}/')
    
    os.system(f'aws s3 sync ./{root}-tiles/ ' + 
              f' s3://grizli-v2/ClusterTiles/Map/{field}/ ' + 
               '--acl public-read --quiet')
                      
    ### Gzip products
    drz_files = glob.glob(f'{root}-*_dr*fits')
    drz_files += glob.glob(f'{root}*seg.fits')
    drz_files.sort()
        
    for file in drz_files:
        cmd = f'gzip --force {file}'
        print(cmd)
        os.system(cmd)
        
    os.system(f'aws s3 sync ./ s3://grizli-v2/ClusterTiles/{field}/' + 
              f' --exclude "*" --include "{root}*gz" --include "*wcs.csv"' + 
              f' --include "*fp.png"')
    
    db.execute(f"update combined_tiles set status=2 where tile = '{tile}' AND field = '{field}'")

    if cleanup:
        print(f'rm -rf {root}-tiles')
        os.system(f'rm -rf {root}-tiles')
        
        files = glob.glob(f'{root}*')
        files.sort()
        for file in files:
            print(f'rm {file}')
            os.remove(file)
        



def get_random_tile():
    """
    Find a visit that needs processing
    """
    from grizli.aws import db
    
    all_tiles = db.SQL(f"""SELECT tile, field 
                           FROM combined_tiles
                           WHERE status=0""")
    
    if len(all_tiles) == 0:
        return None, None
    
    tile, field = all_tiles[np.random.randint(0, len(all_tiles))]
    return tile, field


def run_one():
    """
    Run a single random visit
    """
    import os
    import time
    from grizli.aws import db

    ntile = db.SQL("""SELECT count(status)
                       FROM combined_tiles
                       WHERE status = 0""")['count'][0] 
    
    tile, field = get_random_tile()
    if tile is None:
        with open('/GrizliImaging/tile_finished.txt','w') as fp:
            fp.write(time.ctime() + '\n')
    else:
        print(f'============  Run cosmos tile  ==============')
        print(f'{tile}')
        print(f'========= {time.ctime()} ==========')
        
        with open('/GrizliImaging/tile_history.txt','a') as fp:
            fp.write(f'{time.ctime()} {tile}\n')
        
        #process_visit(tile, clean=clean, sync=sync)
        process_tile(tile=tile, field=field)


def create_mosaic_from_tiles(assoc, filt='ir', clean=True):
    """
    Get tiles overlapping visit footprint
    """
    import glob
    
    import astropy.io.fits as pyfits
    import astropy.table
    
    olap_tiles = db.SQL(f"""SELECT tile, field
                        FROM combined_tiles t, exposure_files e
                        WHERE e.assoc = '{assoc}'
                        AND polygon(e.footprint) && polygon(t.footprint)
                        """)
    
    tx = np.array([int(t.split('.')[0]) for t in olap_tiles['tile']])
    ty = np.array([int(t.split('.')[1]) for t in olap_tiles['tile']])
    
    xm = tx.min()
    ym = ty.min()
    
    nx = tx.max()-tx.min()+1
    ny = ty.max()-ty.min()+1
    
    field = olap_tiles['field'][0]
    for t in olap_tiles['tile']:
        os.system(f"""aws s3 sync s3://grizli-v2/ClusterTiles/{field}/ ./
                          --exclude "*"
                          --include "{field}*{t}-{filt}*_dr?*fits.gz"
                          --include "{field}*{t}-ir_seg.fits.gz"
                          """.replace('\n', ' '))
        
    wcs = db.SQL(f"""SELECT * FROM combined_tiles 
                     WHERE tile = '{xm:02d}.{ym:02d}'
                     AND field = '{field}'""")
    
    h = pyfits.Header()
    for c in wcs.colnames:
        if c in ['footprint']:
            continue
        
        h[c] = wcs[c][0]
    
    xnpix = h['NAXIS1']
    ynpix = h['NAXIS2']
    
    h['NAXIS1'] *= nx
    h['NAXIS2'] *= ny
    
    img_shape = (h['NAXIS2'], h['NAXIS1'])
    sci = np.zeros(img_shape, dtype=np.float32)
    wht = np.zeros(img_shape, dtype=np.float32)
    seg = np.zeros(img_shape, dtype=int)
    
    for tile, txi, tyi in zip(olap_tiles['tile'], tx, ty):
        _file = f'{field}-080-{txi:02d}.{tyi:02d}-{filt}_dr*_sci.fits.gz'
        _files = glob.glob(_file)
        if len(_files) == 0:
            msg = f'*ERROR* {_file} not found'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
            continue
            
        file = _files[0]
        
        msg = f'Add tile {file} to {assoc} mosaic'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        im = pyfits.open(file)
        slx = slice((txi-xm)*xnpix, (txi-xm+1)*xnpix)
        sly = slice((tyi-ym)*ynpix, (tyi-ym+1)*ynpix)      
        
        for k in im[0].header:
            if k not in h:
                h[k] = im[0].header[k]
                
        sci[sly, slx] += im[0].data

        im = pyfits.open(file.replace('_sci','_wht'))
        wht[sly, slx] += im[0].data

        im = pyfits.open(file.replace('_drz_sci','_seg'))
        seg[sly, slx] += im[0].data
    
    _hdu = pyfits.PrimaryHDU(data=sci, header=h)
    _hdu.writeto(f'{assoc}-{filt}_drz_sci.fits', overwrite=True)
    
    _hdu = pyfits.PrimaryHDU(data=wht, header=h)
    _hdu.writeto(f'{assoc}-{filt}_drz_wht.fits', overwrite=True)
    
    _hdu = pyfits.PrimaryHDU(data=seg, header=h)
    _hdu.writeto(f'{assoc}-ir_seg.fits', overwrite=True)
    
    if clean:
        for tile, txi, tyi in zip(olap_tiles['tile'], tx, ty):
            files = glob.glob(f'cos-080-{txi:02d}.{tyi:02d}-{filt}*')
            for file in files:
                msg = f'rm {file}'
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                os.remove(file)

    #### Catalog
    cols = ['id','thresh', 'npix', 'tnpix',
            'xminpix as xmin', 'xmaxpix as xmax',
            'yminpix as ymin', 'ymaxpix as ymax', 
            'x', 'y', 'x2_image', 'y2_image', 'xy_image',
            'errx2', 'erry2', 'errxy',
            'a_image', 'b_image', 'theta_image',
            'cxx_image', 'cyy_image', 'cxy_image',
            'cflux', 'flux',
            'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak',
            'flag',
            'x_image', 'y_image',
            #'x_world', 'y_world',
            'ra as x_world', 'dec as y_world', 
            'flux_iso', 'fluxerr_iso', 'area_iso', 'mag_iso', 'kron_radius',
            'kron_rcirc', 'flux_auto', 'fluxerr_auto', 'bkg_auto',
            'flag_auto', 'area_auto', 'flux_radius_flag', 'flux_radius_20',
            'flux_radius', 'flux_radius_90', 'tot_corr', 'mag_auto',
            'magerr_auto', 'flux_aper_0', 'fluxerr_aper_0', 'flag_aper_0',
            'bkg_aper_0', 'mask_aper_0', 'flux_aper_1', 'fluxerr_aper_1',
            'flag_aper_1', 'bkg_aper_1', 'mask_aper_1', 'flux_aper_2',
            'fluxerr_aper_2', 'flag_aper_2', 'bkg_aper_2', 'mask_aper_2',
            'flux_aper_3', 'fluxerr_aper_3', 'flag_aper_3', 'bkg_aper_3',
            'mask_aper_3']
    
    scols = ','.join(cols)
    
    tabs = []
    for tile, txi, tyi in zip(olap_tiles['tile'], tx, ty):
        
        _SQL = f"SELECT {scols} from combined_tile_phot where tile = '{tile}'"
        tab = db.SQL(_SQL)
                
        # Pixel offsets
        dx = (txi-xm)*4096
        dy = (tyi-ym)*4096
        
        for c in tab.colnames:
            if c in ['xmin', 'xmax', 'x', 'x_image', 'xpeak', 'xcpeak']:
                tab[c] += dx
            elif c in ['ymin', 'ymax', 'y', 'y_image', 'ypeak', 'ycpeak']:
                tab[c] += dy
        
        tabs.append(tab)
        
        msg = f'{assoc}: Query {tile} catalog N={len(tab)} dx={dx} dy={dy}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
    
    tab = astropy.table.vstack(tabs)
    for c in tab.colnames:
        tab.rename_column(c, c.upper())
    
    # Make it look as expected for grizli model
    tab['MAG_AUTO'] = tab['MAG_AUTO'].filled(99)
    tab.rename_column('ID','NUMBER')
    tab.write(f'{assoc}-ir.cat.fits', overwrite=True)
    
    ### Grism
    if 0:
        os.system(f'aws s3 sync s3://grizli-v2/HST/Pipeline/{assoc}/Prep/ ./ --exclude "*" --include "*flt.fits" --include "*yml"')
    
        # grism_files = glob.glob('iehn*[a-p]_flt.fits')
        # grism_files.sort()
        #     
        # grp = auto_script.grism_prep(field_root=assoc, 
        #                              gris_ref_filters={'G141':['ir']},
        #                              files=grism_files,
        #                              refine_mag_limits=[18,23], 
        #                              PREP_PATH='./')
        # 
        # if len(glob.glob(f'{assoc}*_grism*fits*')) == 0:
        #     grism_files = glob.glob('*GrismFLT.fits')
        #     grism_files.sort()
        # 
        #     catalog = glob.glob(f'{assoc}-*.cat.fits')[0]
        #     try:
        #         seg_file = glob.glob(f'{assoc}-*_seg.fits')[0]
        #     except:
        #         seg_file = None
        # 
        #     grp = multifit.GroupFLT(grism_files=grism_files, direct_files=[], 
        #                             ref_file=None, seg_file=seg_file, 
        #                             catalog=catalog, cpu_count=-1, sci_extn=1, 
        #                             pad=256)
        # 
        #     # Make drizzle model images
        #     grp.drizzle_grism_models(root=assoc, kernel='point', scale=0.15)
        # 
        #     # Free grp object
        #     del(grp)
        # 
        # pline = auto_script.DITHERED_PLINE.copy()
        # args_file = f'{assoc}_fit_args.npy'
        # 
        # if (not os.path.exists(args_file)):
        # 
        #     msg = '# generate_fit_params: ' + args_file
        #     utils.log_comment(utils.LOGFILE, msg, verbose=True, show_date=True)
        # 
        #     pline['pixscale'] = 0.1 #mosaic_args['wcs_params']['pixel_scale']
        #     pline['pixfrac'] = 0.5  #mosaic_args['mosaic_pixfrac']
        #     if pline['pixfrac'] > 0:
        #         pline['kernel'] = 'square'
        #     else:
        #         pline['kernel'] = 'point'
        # 
        #     min_sens = 1.e-4
        #     min_mask = 1.e-4
        # 
        #     fit_trace_shift = True
        # 
        #     args = auto_script.generate_fit_params(field_root=assoc, prior=None, MW_EBV=0.0, pline=pline, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, min_sens=min_sens, min_mask=min_mask, sys_err=0.03, fcontam=0.2, zr=[0.05, 3.4], save_file=args_file, fit_trace_shift=fit_trace_shift, include_photometry=False, use_phot_obj=False)
        # 
        #     os.system(f'cp {args_file} fit_args.npy')


def redo_model_from_mosaic(assoc, **kwargs):
    """
    """
    from grizli import utils
    
    import wfc3dash.grism.grism
    import wfc3dash.grism.tiles
    from grizli.aws import db, visit_processor
    
    # if 0:
    #     assoc = 'j100028p0215_0417_dk1_id581181_wfc3ir_f160w-g141'
    
    os.chdir(wfc3dash.grism.grism.HOME_PATH)
    
    visit_processor.update_assoc_status(assoc, status=21)
    
    os.system(f'aws s3 rm --recursive s3://grizli-v2/HST/Pipeline/{assoc} --exclude "*" --include "Extractions/*" --include "Prep/*GrismFLT*" --include "Prep/{assoc}-ir*" ')
    
    os.system(f'aws s3 sync s3://grizli-v2/HST/Pipeline/{assoc}/ ./{assoc} --exclude "*" --include "Prep/*flt.fits"')
    
    if not os.path.exists(f'{assoc}/Extractions'):
        os.mkdir(f'{assoc}/Extractions')
    
    utils.LOGFILE = os.path.join(wfc3dash.grism.grism.HOME_PATH, assoc, 
                                 'Extractions', 
                                 assoc + '_grism.log.txt')
                                     
    os.chdir(f'{assoc}/Prep')
    
    wfc3dash.grism.grism.compute_grism_contamination(assoc, **kwargs)
    
    wfc3dash.grism.grism.sync_products(assoc)
    
    visit_processor.update_assoc_status(assoc, status=22)


def check_phot():
    """
    Red sources are partly from wcs problems
    """
    from grizli.aws import db
    
    ph = db.SQL("""
select p.tile, mag_auto, flu
_radius, flux_auto, flux_aper_1, f105w_tot_corr, f125w_tot_corr, f140w_tot_corr, f160w_tot_corr, f105w_flux_aper_1, f105w_fluxerr_aper_1, f125w_flux_aper_1, f125w_fluxerr_aper_1, f140w_flux_aper_1, f140w_fluxerr_aper_1, f160w_flux_aper_1, f160w_fluxerr_aper_1, f814w_fluxerr_aper_1, f814w_flux_aper_1, f850lp_flux_aper_1, f850lp_fluxerr_aper_1, f850lp_tot_corr, f606w_fluxerr_aper_1, f435w_fluxerr_aper_1, id, ra, dec
    FROM combined_tile_phot p, combined_tiles t
    WHERE p.tile = t.tile AND p.field = t.field
    AND status = 2
    AND mag_auto > 18 and mag_auto < 24 
    """)
    
    hband = 'f160w'
    iband = 'f814w'
    
    ph['hmag'] = 23.9-2.5*np.log10(ph[f'{hband}_flux_aper_1']*ph['flux_auto']/ph['flux_aper_1']*ph[f'{hband}_tot_corr'])
    
    ph['ilim'] = np.maximum(ph[f'{iband}_fluxerr_aper_1']*2, ph[f'{iband}_flux_aper_1'])
    ph['hlim'] = np.maximum(ph[f'{hband}_fluxerr_aper_1']*2, ph[f'{hband}_flux_aper_1'])
    
    ph['ih'] = -2.5*np.log10(ph['ilim']/ph['hlim'])
    
    ph['hmag'].format = '.2f'
    ph['ih'].format = '.2f'
    
    sel = (ph['hmag'] < 24) & (ph['ih'] > 1.0) & (ph['hmag'] > 20)
    sel |= (ph['hmag'] < 24) & (ph['ih'] < -2) & (ph['hmag'] > 20)
    sel = sel.filled(False)
    
    sel &= ~ph['f125w_flux_aper_1'].mask
    
    print(sel.sum())
    
    sub = ph[sel]['tile', 'id','ra','dec','hmag','ih']

    sub['olap'] = [f'<and href="https://grizli-cutout.herokuapp.com/overlap?filters={iband},{hband}&coords={ra}%20{dec}&size=12&mode=files" /> {tile} {id} </a>' for tile, id, ra, dec in zip(sub['tile'], sub['id'], sub['ra'], sub['dec'])]
    
    sub['img'] = [f'<a href="https://grizli-cutout.herokuapp.com/thumb?filters={iband},{hband}&coords={ra}%20{dec}&size=30" /> <img src="https://grizli-cutout.herokuapp.com/thumb?filters={iband},{hband}&coords={ra}%20{dec}&size=12" height=230px> </a>' for ra, dec in zip(sub['ra'], sub['dec'])]


    sub['img'] = [f'<a href="https://grizli-cutout.herokuapp.com/thumb?coords={ra}%20{dec}&size=30" /> <img src="https://grizli-cutout.herokuapp.com/thumb?coords={ra}%20{dec}&size=12" height=230px> </a>' for ra, dec in zip(sub['ra'], sub['dec'])]

    sub['tile', 'olap', 'ra','dec','hmag','ih','img'].write_sortable_html('test.html', localhost=False, max_lines=5000)

def compare_to_deeper():
    """
    Compare extractions in DASH and a pointed visit
    """
    from grizli import multifit
    import matplotlib.pyplot as plt
    
    deep_visit = 'j100028p0215_0417_dk1_id581181_wfc3ir_f160w-g141'
    dash_visit = 'j100028p0215_0651_ehn_cosmos-g141-169_wfc3ir_g141'
    
    args0 = np.load(f'{deep_visit}_fit_args.npy', allow_pickle=True)[0]
    args1 = np.load(f'{dash_visit}_fit_args.npy', allow_pickle=True)[0]
    args0['min_mask'] = 0.000
    args1['min_mask'] = 0.000

    args0['min_sens'] = 0.000
    args1['min_sens'] = 0.000
    
    id = 13046
    #id = 115672
    
    mb0 = multifit.MultiBeam(f'{deep_visit}_{id:05d}.beams.fits', **args0)
    mb1 = multifit.MultiBeam(f'{dash_visit}_{id:05d}.beams.fits', **args1)
    
    tfit0 = mb0.template_at_z(0.6091, templates=args0['t1'])
    tfit1 = mb1.template_at_z(0.6091, templates=args0['t1'])
    
    s0 = mb0.oned_spectrum(tfit=tfit0)['G141']
    s1 = mb1.oned_spectrum(tfit=tfit1)['G141']
    
    plt.plot(s0['wave'], s0['flux'])
    plt.plot(s1['wave'], s1['flux'])
    
    plt.plot(s0['wave'], s0['cont'])
    plt.plot(s1['wave'], s1['cont'])
    
    if 0:
        plt.plot(s0['wave'], s0['cont'] + np.random.normal(size=len(s0))*s0['err'])
        plt.plot(s1['wave'], s1['cont'] + np.random.normal(size=len(s1))*s1['err'])
    
    plt.plot(s0['wave'], s0['err'])
    plt.plot(s1['wave'], s1['err'])
    
    plt.plot(s0['wave'], s0['flux']/s0['err'])
    plt.plot(s1['wave'], s1['flux']/s1['err'])
    
    