# wfc3dash
Helper scripts for WFC3 DASH observations

### Installation

Download and instll the code repositories.  In the shell command line: 

```bash
# Separate repository for manipulating raws
git clone https://github.com/gbrammer/reprocess_wfc3.git
cd reprocess_wfc3
python setup.py install
    
# This repository
git clone https://github.com/gbrammer/wfc3dash.git
cd wfc3dash
python setup.py install
```

### Usage

1) First fetch (only) the raw files of a DASHed exposure

    For example, `icxe01ucq_raw.fits`, from [GO-14114](http://archive.stsci.edu/cgi-bin/mastpreview?mission=hst&dataid=ICXE01010).

2) Run the pre-processing script in the directory with the raw files

```python
>>> from wfc3dash import process_raw
>>> process_raw.run_all()
```

This splits the MultiAccum ramp into individual frames that can be used as individual "FLT" exposures, in this example `icxe01uc[a-j]_flt.fits`.  Note that the `run_all` script also produces the normal `calwf3` IMA and FLT files, `icxe01ucq_ima.fits` and `icxe01ucq_flt.fits`.  The latter will show clearly elongated PSFs for exposures with significant drifts.  Note that they will look different than the FLT files provided by MAST becuase they were produced turning off the `calwf3` CRCORR processing step, which will reject pixels in sources in unpredictable ways as it drifts throughout the ramp.

3) Process the separate read-level exposures (`icxe01uc[a-j]_flt.fits`) as you would a normal FLT, with, e.g., TweakReg and AstroDrizzle.



