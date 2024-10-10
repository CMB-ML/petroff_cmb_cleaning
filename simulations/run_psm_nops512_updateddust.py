#!/usr/bin/env python3

import csv
import random
import os
import subprocess
import time
from pathlib import Path
import joblib

# Path to Healpix directory
HEALPIX_PATH = '/home/petroff/PSM/Healpix_3.50'

# Path to Healpy binary directory, e.g., where anafast_cxx executable is
HEALPY_BIN_PATH = '/home/petroff/PSM/Healpix_3.50/src/healpy/build/temp.linux-x86_64-2.7/bin'

# Path to CLASS code directory, e.g., where class executable is
CLASS_CODE_PATH = '/home/petroff/PSM/PSM-v1_7_8/Soft/libraries/class'

# Path to WMAP9 LCDM chains directory
CHAINS_BASE = Path('/home/petroff/PSM/wmap_lcdm_wmap9_chains_v5')

# Set up IDL paths
PRE_SCRIPT = '''
DEFSYSV, '!PSMROOT', '/home/petroff/PSM/PSM-v1_7_8'
!path=!path+':'+expand_path('+'+!PSMROOT+'/Soft/')+':/home/petroff/PSM/astrolib/pro:/home/petroff/PSM/Healpix_3.50/src/idl/toolkit:/home/petroff/PSM/Healpix_3.50/src/idl/misc:/home/petroff/PSM/Healpix_3.50/src/idl/fits:/home/petroff/PSM/Healpix_3.50/src/idl/interfaces:/home/petroff/PSM/Healpix_3.50/src/idl/visu:/home/petroff/PSM/Healpix_3.50/src/idl/zzz_external/obsolete_astron:/home/petroff/PSM/mpfit:/home/petroff/PSM/coyote'
'''

# Number of simulations to run
NUM_SIMS = 1000

# Output base path
BASE_PATH = Path('/home/petroff/PSM/outputnops512_updateddust')


os.environ['PATH'] = f'{HEALPY_BIN_PATH}:{CLASS_CODE_PATH}:' + os.environ['PATH']
os.environ['HEALPIX'] = HEALPIX_PATH

CONFIG = '''
#
# Set output directory, seed, and simulate polarization
#

OUTPUT_DIRECTORY = {path}/PSM_OUTPUT_{seed}
CLEAR_ALL = yes
SEED = {seed}
FIELDS = T
VISU = 0
OUTPUT_VISU = ps



#
# Set resolution and pixelization
#

SKY_RESOLUTION = 13.1
SKY_LMAX = 1536
SKY_PIXELISATION = HEALPIX
HEALPIX_NSIDE = 512



#
# Only simulate galactic foregrounds
#

DIPOLE_MODEL = no_dipole
CMB_MODEL = gaussian
SZ_MODEL = nbody+hydro
GAL_MODEL = simulation
PS_MODEL = no_ps
FIRB_MODEL = jgn2005



#
# Configure CMB simulation
#

RUN_CLASS = yes
CMB_CL_SOURCE = CLASS
CMB_LENSING = cl



#
# Constrain SZ
#

SZ_CONSTRAINED = yes



#
# Cosmological Parameters (vanilla LCDM)
# T_CMB, HE_FRACTION, and N_MASSLESS_NU set to defaults
#

T_CMB = 2.725
H = {H0}
OMEGA_M = {omegam}
OMEGA_B = {omegab}
OMEGA_NU = 0
OMEGA_K = 0
SIGMA_8 = {sigma8}
N_S = {ns002}
N_S_RUNNING = 0
N_T = 0
R = 0
TAU_REION = {tau}
HE_FRACTION = 0.24
N_MASSLESS_NU = 3.04
N_MASSIVE_NU = 0
W_DARK_ENERGY = -1
KPIVOT = 0.002
SCALARAMPLITUDE = {a002}



#
# Use all diffuse galactic emission components
#

INCLUDE_SYNCHROTRON = yes
INCLUDE_FREEFREE = yes
INCLUDE_SPINDUST = yes
INCLUDE_THERMALDUST = yes
INCLUDE_CO = yes



#
# Use 2018 RIMO
#

OBS_TASK = new
WHAT_OBS = fullobs
OBS_RES = sky
OBS_COADD = allsky

INSTRUMENT = HFI_RIMO
INSTRUMENT = LFI_RIMO

HFI_RIMO_VERSION = R3.00
LFI_RIMO_VERSION = R3.31

HFI_RIMO_100_DET = F100
HFI_RIMO_143_DET = F143
HFI_RIMO_217_DET = F217
HFI_RIMO_353_DET = F353
HFI_RIMO_545_DET = F545
HFI_RIMO_857_DET = F857

LFI_RIMO_070_DET = F070
'''

SCRIPT = PRE_SCRIPT + '''
PSM_MAIN, '{}'
exit
'''


# Parameters from WMAP9 chains
# https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_wmap9_chains_v5.tar.gz
random.seed(1296570)
chain_idx = [random.randrange(1296570) for _ in range(NUM_SIMS)]
params = ['H0', 'omegam', 'omegab', 'sigma8', 'ns002', 'tau', 'a002']
param_vals = {}
for param in params:
    with open(CHAINS_BASE / param) as infile:
        reader = csv.reader(infile, delimiter=' ', skipinitialspace=True)
        param_vals[param] = [float(row[1]) for idx, row in enumerate(reader) if idx in chain_idx]


def run_sim(seed):
    # Run PSM
    config_file = BASE_PATH / f'config_{seed}.psm'
    config_params = {param: param_vals[param][seed] for param in params}
    config_params['seed'] = seed
    config_params['path'] = BASE_PATH
    config_params['H0'] /= 100.
    config_params['a002'] /= 1e9
    config_file.write_text(CONFIG.format(**config_params))
    subprocess.run(['idl'], input=SCRIPT.format(config_file).encode())
    config_file.unlink()
    
    # Clean up output
    (BASE_PATH / f'sim{seed:04d}').mkdir()
    dets = {
        'LFI': ['070'],
        'HFI': ['100', '143', '217', '353', '545', '857'],
    }
    for band in dets:
        for det in dets[band]:
            (BASE_PATH / f'PSM_OUTPUT_{seed}/observations/{band}_RIMO'
             / f'detector_F{det}/group1_map_detector_F{det}.fits').rename( \
            BASE_PATH / f'sim{seed:04d}/{det}.fits')
    (BASE_PATH / f'PSM_OUTPUT_{seed}/components/cmb/cmb_map.fits').rename( \
        BASE_PATH / f'sim{seed:04d}/cmb.fits')
    (BASE_PATH / f'PSM_OUTPUT_{seed}/components/cmb/cmb_lensed_cl.fits').rename( \
        BASE_PATH / f'sim{seed:04d}/cmb_lensed_cl.fits')
    subprocess.run(['rm', '-r', BASE_PATH / f'PSM_OUTPUT_{seed}'])

failed_seeds = []

def try_run_sim(seed):
    try:
        run_sim(seed)
    except:
        failed_seeds.append(seed)

os.mkdir(BASE_PATH)
start_time = time.time()
joblib.Parallel(n_jobs=40)(joblib.delayed(try_run_sim)(seed) for seed in range(NUM_SIMS))

print('Failed seeds:', failed_seeds)
print(f'Ran {NUM_SIMS} simulations in {time.time() - start_time:.1f} seconds')
