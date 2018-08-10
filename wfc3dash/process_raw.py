"""
Split ramps into individual FLT exposures. 

To use, download *just* the RAW files for a given visit/program.

    >>> from wfc3dash import process_raw
    >>> process_raw.run_all()

"""

def run_all(skip_first_read=True):
    """
    Run splitting script on all RAW files in the working directory.  
    
    First generates IMA files from RAWs after setting CRCORR=OMIT.
    
    """
    
    import os
    import glob
    import astropy.io.fits as pyfits
    
    import wfc3tools
    
    files = glob.glob("*raw.fits")
    files.sort()
    
    for file in files:
        if os.path.exists(file.replace('_raw','_ima')):
            print('IMA exists, skip', file)
            continue
        
        print('Process', file)
        
        # Set CRCORR 
        raw_im = pyfits.open(file, mode='update')
        raw_im[0].header['CRCORR'] = 'OMIT'
        raw_im.flush()

        # Remove FLT if it exists or calwf3 will die
        if os.path.exists(file.replace('_raw','_flt')):
            os.remove(file.replace('_raw','_flt'))
        
        # Run calwf3
        wfc3tools.calwf3(file)
        
        # Split into individual FLTs
        split_ima_flt(file=file.replace('_raw','_ima'), skip_first_read=skip_first_read)
        
def split_ima_flt(file='icxe15x0q_ima.fits', skip_first_read=True):
    """
    Pull out reads of an IMA file into individual "FLT" files
    """
    import os
    import astropy.io.fits as pyfits
    import numpy as np
    
    # https://github.com/gbrammer/reprocess_wfc3
    from reprocess_wfc3.reprocess_wfc3 import get_flat, split_multiaccum
    
    ima = pyfits.open(file)
    root = file.split("_ima")[0][:-1]
    
    #### Pull out the data cube, order in the more natural sense
    #### of first reads first
    cube, dq, time, NSAMP = split_multiaccum(ima, scale_flat=False)
    
    #### Readnoise in 4 amps
    readnoise_2D = np.zeros((1024,1024))
    readnoise_2D[512: ,0:512] += ima[0].header['READNSEA']
    readnoise_2D[0:512,0:512] += ima[0].header['READNSEB']
    readnoise_2D[0:512, 512:] += ima[0].header['READNSEC']
    readnoise_2D[512: , 512:] += ima[0].header['READNSED']
    readnoise_2D = readnoise_2D**2

    #### Gain in 4 amps
    gain_2D = np.zeros((1024,1024))
    gain_2D[512: ,0:512] += ima[0].header['ATODGNA']
    gain_2D[0:512,0:512] += ima[0].header['ATODGNB']
    gain_2D[0:512, 512:] += ima[0].header['ATODGNC']
    gain_2D[512: , 512:] += ima[0].header['ATODGND']
    
    #### Need to put dark back in for Poisson
    dark_file = ima[0].header['DARKFILE'].replace('iref$', os.getenv('iref')+'/')
    dark = pyfits.open(dark_file)
    dark_cube, dark_dq, dark_time, dark_NSAMP = split_multiaccum(dark, scale_flat=False)
    
    #### Need flat for Poisson
    if ima[0].header['FLATCORR'] == 'COMPLETE':
        flat_im, flat = get_flat(ima)
    else:
        flat_im, flat = None, 1
        
    #### Subtract diffs of flagged reads
    diff = np.diff(cube, axis=0)
    dark_diff = np.diff(dark_cube, axis=0)
    dt = np.diff(time)
        
    final_sci = diff
    final_dark = dark_diff[:NSAMP-1]
    final_exptime = dt
    
    final_var = final_sci*0
    final_err = final_sci*0
    for i in range(NSAMP-1):
        final_var[i,:,:] = readnoise_2D + (final_sci[i,:,:]*flat + final_dark[i,:,:]*gain_2D)*(gain_2D/2.368) 
        if ima[0].header['FLATCORR'] == 'COMPLETE':
            final_var[i,:,:] += (final_sci[i,:,:]*flat*flat_im['ERR'].data)**2
        
        final_err[i,:,:] = np.sqrt(final_var[i,:,:])/flat/(gain_2D/2.368)/1.003448/final_exptime[i]
        final_sci[i,:,:] /= final_exptime[i]
    
    h_0 = ima[0].header.copy()
    h_sci = ima['SCI',1].header.copy()
    h_err = ima['ERR',1].header.copy()
    h_dq = ima['DQ',1].header.copy()
    h_time = ima['TIME',1].header.copy()
    
    final_dq = dq[1:,:,:]*1
    final_dq -= (final_dq & 2048)
    
    h_sci['CRPIX1'] = 507
    h_sci['CRPIX2'] = 507
    letters = 'abcdefghijklmno'
    for i in range(1*skip_first_read,NSAMP-1):
        h_0['EXPTIME'] = final_exptime[i]
        h_0['IREAD'] = i
        hdu = pyfits.HDUList(pyfits.PrimaryHDU(header=h_0))
        
        hdu.append(pyfits.ImageHDU(data=final_sci[i,5:-5,5:-5], header=h_sci))
        hdu.append(pyfits.ImageHDU(data=final_err[i,5:-5,5:-5], header=h_err))
        hdu.append(pyfits.ImageHDU(data=final_dq[i,5:-5,5:-5], header=h_dq))
        h_time['PIXVALUE'] = final_exptime[i]
        h_time['NPIX1'] = 1014
        h_time['NPIX2'] = 1014
        
        hdu.append(pyfits.ImageHDU(header=h_time))
        hdu.writeto('%s%s_flt.fits' %(root, letters[i-1]), clobber=True)
        print('%s%s_flt.fits' %(root, letters[i-1]))
