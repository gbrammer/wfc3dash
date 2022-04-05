"""
Tiles in COSMOS
"""
import os
import glob
import numpy as np

def make_cosmos_tiles():
    """
    Make tiles across COSMOS with a single tangent point
    """
    import numpy as np
    
    from grizli_aws import field_tiles
    from grizli.aws import tile_mosaic
    from grizli import utils
    from grizli.aws import db
    
    np.set_printoptions(precision=6)
    
    tile_npix = 4096
    pscale = 0.1
    tile_arcmin = tile_npix*pscale/60
    print('Tile size, arcmin = ', tile_arcmin)
    
    ra, dec = 150.1, 2.29
    rsize = 48
    
    name='cos'
    tiles= field_tiles.define_tiles(ra=ra, dec=dec, 
                                           size=(rsize*2, rsize*2), 
                                           tile_size=tile_arcmin, 
                                           overlap=0, field=name, 
                                           pixscale=pscale, theta=0)
    
    filt = 'G141'
    extra = " AND file like 'iehn%%' "

    filt = 'F814W'
    extra = ""
    
    fig, tab = tile_mosaic.exposure_map(ra, dec, rsize, name, 
                                        filt=filt.upper(), s0=18, 
                                        extra=extra)
    fig.tight_layout(pad=0.5)
    fig.savefig('/tmp/map.png')
    fig.tight_layout(pad=0.5)
    fig.savefig('/tmp/map.png')
    
    ax = fig.axes[0]
    rows = []
    
    for t in tiles:
        wcs = tiles[t]
        sr = utils.SRegion(wcs.calc_footprint())
        ax.add_patch(sr.patch(fc='None', ec='r', alpha=0.5, 
                              transform=ax.get_transform('world'))[0])
        
        ax.text(*sr.centroid[0], t, ha='center', va='center', 
                transform=ax.get_transform('world'), fontsize=7)
                
        wcsh = utils.to_header(wcs)
        row = [t]
        for k in wcsh:
            row.append(wcsh[k])
        
        fp = sr.xy[0].__str__().replace('\n', ',').replace('  ', ',').replace(' ', '').replace('[','(').replace(']', ')')
        row.append(fp)     
        
        rows.append(row)
    
    names = ['tile']
    for k in wcsh:
        names.append(k.lower())
    
    names.append('footprint')
    
    cos_tile = utils.GTable(rows=rows, names=names)
    cos_tile['status'] = 0
    
    db.send_to_database('cosmos_tiles', cos_tile, if_exists='replace')


def process_tile(tile='01.01', filters=['F350LP', 'F435W', 'F475W', 'F606W', 'F775W', 'F814W', 'F850LP', 'F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']):
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
    
    db.execute(f"update cosmos_tiles set status=1 where tile = '{tile}'")
    
    row = db.SQL(f"select * from cosmos_tiles where tile = '{tile}'")
    h = pyfits.Header()
    for k in row.colnames:
        if k in ['footprint']:
            continue
        
        h[k.upper()] = row[k][0]
        
    ir_wcs = pywcs.WCS(h)
    
    root = f'cos-tile-{tile}'
    
    visit_processor.cutout_mosaic(rootname=root, 
                              skip_existing=False,
                              ir_wcs=ir_wcs,
                              filters=filters, 
                              kernel='square', s3output=None, 
                              gzip_output=False, clean_flt=True)
    
    files = glob.glob(f'{root}*dr*fits')
    if len(files) == 0:
        db.execute(f"update cosmos_tiles set status=4 where tile = '{tile}'")
        return True
        
    golfir.catalog.make_charge_detection(root, ext='ir')
    
    phot = auto_script.multiband_catalog(field_root=root) #, **phot_kwargs)
    # for c in ['number']:
    #     if c in phot.colnames:
    #         phot.remove_column(c)
    
    for i in [4,5,6]:
        for c in phot.colnames:
            if c.endswith(f'{i}'):
                phot.remove_column(c)
    
    for c in phot.colnames:
        if c in ['ra','dec']:
            continue
            
        if phot[c].dtype == np.float64:
            phot[c] = phot[c].astype(np.float32)
        elif phot[c].dtype == np.int64:
            phot[c] = phot[c].astype(np.int32)
              
    phot['tile'] = tile
    
    if 'id' in phot.colnames:
        phot.remove_column('id')
    
    for c in ['xmin','xmax','ymin','ymax']:
        if c in phot.colnames:
            phot.rename_column(c, c+'pix')
    
    db.execute(f"DELETE from cosmos_tile_phot WHERE tile='{tile}'")
    
    seg = pyfits.open(f'{root}-ir_seg.fits')
    
    ### IDs on edge
    edge = np.unique(seg[0].data[0,:])
    edge = np.append(edge, np.unique(seg[0].data[-1,:]))
    edge = np.append(edge, np.unique(seg[0].data[:,0]))
    edge = np.append(edge, np.unique(seg[0].data[:,-1]))
    edge = np.unique(edge)
    phot['edge'] = np.in1d(phot['number'], edge)*0
    
    ### Add missing columns
    cols = db.SQL('select * from cosmos_tile_phot limit 2').colnames
    for c in phot.colnames:
        if c not in cols:
            print('Add column {0} to `cosmos_tile_phot` table'.format(c))
            if phot[c].dtype in [np.float64, np.float32]:
                SQL = "ALTER TABLE cosmos_tile_phot ADD COLUMN {0} real;".format(c)
            else:
                SQL = "ALTER TABLE cosmos_tile_phot ADD COLUMN {0} int;".format(c)
                
            db.execute(SQL)
            
    db.send_to_database('cosmos_tile_phot', phot, if_exists='append')
    
    if 'id' not in cols:
        # Add id column
        db.execute('ALTER TABLE cosmos_tile_phot ADD COLUMN id SERIAL PRIMARY KEY;')
    
    # Use db id
    ids = db.SQL(f"SELECT number, id, ra, dec, tile from cosmos_tile_phot WHERE tile='{tile}'")
    idx, dr = ids.match_to_catalog_sky(phot)
        
    phot['id'] = phot['number']
    for c in ['xmin','xmax','ymin','ymax']:
        if c+'pix' in phot.colnames:
            phot.rename_column(c+'pix', c)
    
    golfir.catalog.switch_segments(seg[0].data, phot, ids['id'][idx])
    pyfits.writeto(f'{root}-ir_seg.fits', data=seg[0].data, 
                   header=seg[0].header, overwrite=True)
    
    drz_files = glob.glob(f'{root}-*_dr*fits')
    drz_files += glob.glob(f'{root}*seg.fits')
    drz_files.sort()
        
    for file in drz_files:
        cmd = f'gzip --force {file}'
        print(cmd)
        os.system(cmd)
        
    os.system(f"""aws s3 sync ./ s3://grizli-v2/HST/Cosmos/Tiles/ --exclude "*" --include "{root}*gz" --include "*wcs.csv" --include "*fp.png" """)
    
    db.execute(f"update cosmos_tiles set status=2 where tile = '{tile}'")


def get_random_tile():
    """
    Find a visit that needs processing
    """
    from grizli.aws import db
    
    all_tiles = db.SQL(f"""SELECT DISTINCT(tile) 
                           FROM cosmos_tiles
                           WHERE status=0""")
    
    if len(all_tiles) == 0:
        return None
    
    random_tile = all_tiles[np.random.randint(0, len(all_tiles))][0]
    return random_tile


def run_one():
    """
    Run a single random visit
    """
    import os
    import time
    from grizli.aws import db

    ntile = db.SQL("""SELECT count(distinct(tile))
                       FROM cosmos_tiles
                       WHERE status = 0""")['count'][0] 
    
    tile = get_random_tile()
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
        process_tile(tile=tile)
