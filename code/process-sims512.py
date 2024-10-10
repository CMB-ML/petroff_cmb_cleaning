import os
from pathlib import Path
import numpy as np
import healpy as hp

# Data locations
PLANCK_MAP_DIR = Path("/data/planck")
SIM_DIR = Path("/data/users/petroff/psm-sims/outputnops512_updateddust")
SIM_OUT_DIR = Path("/data/users/petroff/psm-sims/outputnops512_updateddust_processed")
FIELD = 0

NSIDE = 512
N_TRAIN = 800
N_TEST = 200

np.random.seed(487349857)

#
# Load data
#

planck_map_names = [
    "LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits",
    "HFI_SkyMap_100_2048_R3.01_full.fits",
    "HFI_SkyMap_143_2048_R3.01_full.fits",
    "HFI_SkyMap_217_2048_R3.01_full.fits",
    "HFI_SkyMap_353_2048_R3.01_full.fits",
    "HFI_SkyMap_545_2048_R3.01_full.fits",
    "HFI_SkyMap_857_2048_R3.01_full.fits",
]

# Use TT covariance for noise distribution
planck_noise_maps = []
for pmn in planck_map_names:
    field = FIELD + 2 if pmn[11:14] in ["545", "857"] else FIELD + 4
    planck_noise_maps.append(
        hp.ud_grade(
            np.abs(
                hp.read_map(
                    PLANCK_MAP_DIR / pmn, verbose=False, field=field, dtype=np.float32
                )
            ),
            NSIDE,
            order_out="NESTED",
            power=2,
        )
    )
planck_noise_maps = np.sqrt(planck_noise_maps)

# Determine min / max
x_max = np.zeros(len(planck_map_names))
y_max = 0
for i in range(N_TRAIN + N_TEST):
    y_max = max(
        np.max(
            np.abs(
                hp.read_map(
                    SIM_DIR / f"sim{i:04}/cmb.fits",
                    verbose=False,
                    field=FIELD,
                    dtype=np.float32,
                )
            )
        ),
        y_max,
    )
    for j, freq in enumerate([pmn[11:14] for pmn in planck_map_names]):
        x_max[j] = max(
            np.max(
                np.abs(
                    hp.read_map(
                        SIM_DIR / f"sim{i:04}/{freq}.fits",
                        verbose=False,
                        field=FIELD,
                        dtype=np.float32,
                    )
                )
            ),
            x_max[j],
        )

os.makedirs(SIM_OUT_DIR)
np.savez_compressed(
    SIM_OUT_DIR / "normalization.npz", x_max=x_max, y_max=y_max
)

for i in range(N_TRAIN + N_TEST):
    y_raw = hp.reorder(
        hp.read_map(
            SIM_DIR / f"sim{i:04}/cmb.fits",
            verbose=False,
            field=FIELD,
            dtype=np.float32,
        ),
        r2n=True,
    )
    y_raw = y_raw / y_max * 0.5 + 0.5
    os.makedirs(SIM_OUT_DIR / f"sim{i:04}")
    hp.write_map(SIM_OUT_DIR / f"sim{i:04}/cmb.fits", y_raw, nest=True)
    for j, freq in enumerate([pmn[11:14] for pmn in planck_map_names]):
        x_raw = hp.reorder(
            hp.read_map(
                SIM_DIR / f"sim{i:04}/{freq}.fits",
                verbose=False,
                field=FIELD,
                dtype=np.float32,
            ),
            r2n=True,
        ) + np.random.normal(scale=planck_noise_maps[j]).astype(np.float32)
        x_raw /= x_max[j]
        hp.write_map(SIM_OUT_DIR / f"sim{i:04}/{freq}.fits", x_raw, nest=True)
