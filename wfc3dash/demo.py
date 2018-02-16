
import os
from wfc3dash import process_raw

# https://github.com/gbrammer/grizli/
from grizli import utils, prep
import grizli.ds9

#######################
# Fetch demo data
os.system('wget http://www.stsci.edu/~brammer/WFC3/dash_demo.tar.gz')
os.system('tar xzvf dash_demo.tar.gz')

#######################
# Preprocessing to split ramps into individual files
os.chdir('RAW')
process_raw.run_all()

#######################
# Align, etc.
os.chdir('../Process')

# Make "visit" dictionaries for each eaposure
ima_files=glob.glob('../RAW/*ima.fits')
ima_files.sort()

visits = []
for file in ima_files:
    root=os.path.basename(file).split("_ima")[0][:-1]
    files=glob.glob('../RAW/%s*[a-o]_flt.fits' %(root))
    if len(files) == 0:
        continue
    
    files = [os.path.basename(file) for file in files]
    visit = {'product':root, 'files':files}

    visits.append(visit)

# Manual alignment, shifts can be very large
ds9 = grizli.ds9.DS9()
for visit in visits:
    grizli.prep.manual_alignment(visit, ds9, reference='ipac_acs_bright.reg')

# Astrometric alignment file, simply columns of RA/Dec
radec = 'ipac_acs_galaxy.radec'

# Run the aligment script
for visit in visits:
    status = grizli.prep.process_direct_grism_visit(direct=visit, grism={}, radec=radec, skip_direct=False, align_mag_limits=[14,24], tweak_max_dist=8, tweak_threshold=8, align_tolerance=8, tweak_fit_order=2)

