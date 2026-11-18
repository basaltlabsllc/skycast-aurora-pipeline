# skycast-aurora-pipeline

Turns NOAA SWPC's live solar-wind data into aurora forecast frames for the
SkyCast app, using the OVATION Prime auroral precipitation model.

- `pipeline/solarwind.py` reads the real-time solar wind (RTSW) wind and field
  feeds, keeps the active spacecraft, and computes the Newell coupling with
  OVATION Prime's 4-hour weighting (each hour back weighted 0.65×), shifted by
  the L1-to-Earth delay from SWPC's propagated solar wind.
- `pipeline/model.py` runs OVATION Prime (OvationPyme) for one coupling value
  and resamples diffuse, mono and wave energy flux (erg cm⁻² s⁻¹) and average
  energy (keV) onto a 1° geographic grid with AACGM-v2. Average energy 0 where
  energy flux > 0 means "energy unknown" (OvationPyme's own number-flux
  threshold), not low energy.

- `pipeline/encode.py` packs one model time into an AUR1 frame file;
  `pipeline/site.py` builds every frame of a run plus `manifest.json`;
  `run_aurora.py --out <dir>` fetches the three feeds and writes the site.

## A run

`python run_aurora.py --out site` writes `site/manifest.json` and
`site/frames/<YYYYMMDDTHHMMZ>.aur1`, one frame per 10 minutes from
`runTime − 60 min` (rounded down to the step) to `windowEnd` (the newest solar
wind's arrival at Earth, rounded down). Each frame uses only wind that has
reached Earth by its time; before each frame the four hours it averages are
checked for holes over 15 minutes. One L1→Earth delay, from SWPC's newest
propagated row, is applied to every frame of a run, so a past frame can use a
sample up to about 10 minutes early when the wind speed was rising.

Exit codes; every failure writes nothing (the site is built in memory, written
to a temp dir beside `--out`, and renamed into place only after every frame
succeeded; `--out` must be absent or empty):

| code | meaning |
|---|---|
| 0 | site written |
| 2 | solar-wind gap, wind too stale to reach the window, or an L1→Earth delay that is missing or outside 10–180 min |
| 3 | every frame's energy flux is below the encoder's floor (0.01 erg cm⁻² s⁻¹) in all three types, so every frame would be blank |
| 1 | anything else (a feed could not be fetched, `--out` not empty, …) |

The GitHub Actions workflow (`.github/workflows/aurora.yml`) runs every 30
minutes and can also be started by hand (`workflow_dispatch`). It installs requirements, runs the tests, runs `run_aurora.py --out
site`, copies `colortable.json` into `site/` when the repo has one, and
force-pushes `site/` as an orphan commit to `gh-pages` with the built-in
`GITHUB_TOKEN` (the smoke pipeline's publish step). A failed run skips the
publish, so gh-pages keeps the previous run.

Live run, 2026-11-18T00:55:57Z (this Mac, Python 3.11, fetch included):
13 frames, 771,395 bytes of frames (59,338 per frame on average, 58,931–59,690),
773,383 bytes with the manifest, 10.8 s. Spacecraft SOLAR1, L1→Earth delay
58 min 10 s, dF 4,013–4,606.

## Frame format (AUR1)

A frame file is gzip of: the bytes `AUR1`, uint16 LE width (360), uint16 LE
height (181), then six uint8 planes of width × height in the order
`diffuse.eflux, diffuse.eavg, mono.eflux, mono.eavg, wave.eflux, wave.eavg`;
row 0 is latitude −90, column 0 is longitude −180, 1° cells.

Byte 0 means none. Bytes 1–255 are log-spaced:
`value = lo * (hi/lo) ** ((b-1)/254)`, energy flux over 0.01–100
erg cm⁻² s⁻¹, average energy over 0.05–50 keV (the ranges are also in the
manifest's `enc`). The encoder picks the nearest byte in log space (worst error
1.8 % for flux, 1.4 % for energy); values above the range clamp to 255; values
below it, including 0, become byte 0, so energy flux under 0.01 erg cm⁻² s⁻¹ is
not carried.

**Energy unknown.** A cell can have energy flux > 0 with average energy 0: the
model gave energy flux but OvationPyme's number-flux rules gave no energy.
The encoder keeps that: eflux encodes normally and eavg is byte 0. Read eavg
byte 0 under a lit eflux as "energy unknown", never as low energy.

## manifest.json

`{ v: 1, model, runTime, spacecraft, swNewest, swArrival, windowStart,
windowEnd, stepMin: 10, enc: { eflux: [0.01, 100], eavg: [0.05, 50] },
frames: [{ t, path, swUsedTo }] }`, all times ISO UTC, frames oldest first.
`swNewest` is the L1 time of the newest usable solar-wind sample and
`swArrival` that time plus the L1→Earth delay from SWPC's newest propagated
row. `swUsedTo` is the newest L1 sample a frame used. `spacecraft` is the
source of the **newest** sample: SWPC can switch the active spacecraft
mid-series, in which case older minutes in the same run came from another one.

## Python version

Runs on **Python 3.11** with **setuptools<66** (see requirements.txt).
OvationPyme imports nasaomnireader, whose `default_config.py` calls
`pkg_resources.appdirs` on import. Only setuptools 65.x and older provide it,
and their `pkg_resources` uses `pkgutil.ImpImporter`, which Python 3.12
removed. OvationPyme and nasaomnireader are installed unmodified from upstream,
pinned by full commit hash.

Run the tests from the repo root: `python -m pytest -q`.

## Colour table (GLOW)

`colortable/build_table.py --glow-exe <glow.bin> --out colortable.json` runs
NCAR GLOW once per average energy and spectral shape and writes
`{ energiesKeV[16, 0.1–30 log-spaced], altKm[43: 80–500 step 10], shapes:
{ maxwellian, gaussian, broadband } each { green, red, blue }:
number[energy][alt], shapeNotes, meanEnergyKeV, energyWeightedMeanKeV,
broadbandCharacteristicEnergyKeV, background, precision, source }`: volume emission rates of 557.7, 630.0 and 427.8 nm in
photons cm⁻³ s⁻¹ at an energy flux of 1 erg cm⁻² s⁻¹, counting only the light
the precipitation causes. Conditions: 65° N, 0° E, 00:00 UT (local midnight),
2026 day 079 (equinox), F10.7 = 120, Ap = 15, no solar input. The builder's
header cites the GLOW routines and lines used.

Each value is GLOW's printed output for the precipitation run minus GLOW's
no-precipitation run (same date, place and indices) at each GLOW altitude,
clamped at 0 (a residual below −0.05 photons cm⁻³ s⁻¹ inside 80–500 km
stops the build instead), linearly interpolated onto the 10-km levels and rounded to 4
significant figures. The subtracted airglow's column brightness per line is
in `background` (about 31 R of red, 5 R of green, no blue); left in, it would
scale with energy flux in the app.

**Precision:** GLOW's driver prints emission rates with 2 decimal places, so
the table's absolute resolution is 0.01 photons cm⁻³ s⁻¹; small values in the
profile tails have fewer than 4 real significant figures (`precision`).

Shapes (all made by GLOW's own spectrum routine MAXT and normalised by it to
1 erg cm⁻² s⁻¹ on GLOW's own energy-bin widths; `shapeNotes` says the same in
the file):

- `maxwellian` (diffuse): GLOW's Maxwellian, characteristic energy = axis
  energy / 2, so its mean energy is the axis energy.
- `gaussian` (mono): **GLOW's monoenergetic spectrum, not a Gaussian** (GLOW
  has none): all the flux in the one GLOW energy bin containing the axis
  energy. The key keeps the name the app reads. Mono layers are 7–13 km thick,
  so on the 10-km levels the peak altitude is resolved only to the level; the
  column agrees with GLOW's own grid within 0.5 %.
- `broadband` (wave): **an approximation of a broadband spectrum**: GLOW's
  Maxwellian with its Meier et al. (1989) low-energy tail. The builder bisects
  the characteristic energy until the mean energy of the spectrum GLOW builds
  (printed energy flux ÷ number flux) equals the axis energy within 2 %
  (`meanEnergyKeV`; the characteristic energies used are in
  `broadbandCharacteristicEnergyKeV`). The tail holds many soft electrons, so
  the core is harder than a Maxwellian of the same mean: its energy-weighted
  mean is 2.6–3.7 × the axis energy (`energyWeightedMeanKeV`), against the
  Maxwellian's 1.5 ×. It therefore emits lower and less red than a Maxwellian
  of the same average energy, which is the opposite of a soft sub-keV broadband
  population, so **the app does not use this shape for now**. At 20–30 keV part
  of the broadband blue and green lies below the table's 80 km floor: at 30 keV
  the 80-km blue value is 0.86 of its peak, and at least 13 % of the blue column
  is in 73.5–80 km (GLOW's grid starts there).

At the same axis energy the mono layer is narrower than the Maxwellian's and
peaks higher: energy deposition follows the energy-weighted mean energy, which
for a Maxwellian is about 1.5 × the axis energy, while for mono it is the axis
energy itself. On GLOW's own altitude grid (green
line, background subtracted):

| axis energy | Maxwellian peak / FWHM | mono peak / FWHM |
|---|---|---|
| 3.06 keV | 117.2 km / 26.8 km | 126.8 km / 12.7 km |
| 4.48 keV | 112.6 km / 20.8 km | 119.5 km / 7.2 km |
| 6.56 keV | 108.2 km / 15.6 km | 112.6 km / 6.9 km |

The Maxwellian at 3.06 keV has an energy-weighted mean of 4.60 keV, and the
4.48 keV mono peaks at 119.5 km, near its 117.2 km. The tests check that mono
is never wider than the Maxwellian on the 10-km levels (strictly narrower at
3 keV; at 0.1, 6.6, 20.5 and 30 keV both are a level or two wide and tie) and
that each mono green peak lies within one level of the peak of the Maxwellian
whose energy-weighted mean is closest to its energy.

**Driver patch.** `colortable/glowpython.patch` changes one file,
`src/ncarglow/fortran/glowpython.f90`: it reads optional arguments 12–14 as
MAXT's `itail`, `fmono` and `emono` (the unpatched driver fixes all three at 0)
and also calls MAXT when only `fmono` is given. Nothing else in GLOW changes;
with no extra arguments the output is byte-identical to the unpatched build.

GLOW is GLOW 0.981 as packaged by
[space-physics/ncar-glow](https://github.com/space-physics/ncar-glow)
(package `ncarglow` 1.4.0, not on PyPI) at commit
`2c099fdc2bd02b69fd1c3f36e1cc603ad3d34336`. Only its Fortran driver
(`glow.bin`) is built and run; the Python wrapper looks up F10.7 and Ap
online, and the table fixes both.

The workflow `.github/workflows/colortable.yml` (`workflow_dispatch` only)
checks out that commit, applies the patch, installs gfortran and cmake, builds
`glow.bin`, builds the table, runs `tests/test_colortable.py` and commits
`colortable.json` to `main` with the built-in `GITHUB_TOKEN`; a failing test
stops the commit. It runs only when dispatched from `main`. The tests skip
when `colortable.json` is absent, except with `CI` set, where they fail;
`COLORTABLE=<path>` points them at another file.

## Licences

- **OvationPyme** (OVATION Prime in Python) — LGPL-3.0. Used unmodified, as a
  separate library; this repo only reproduces its solar-wind averaging scheme.
- **GLOW** (airglow/aurora emission model) — research use only.
  - The GLOW Fortran: `src/ncarglow/fortran/Glowlicense.txt` (also
    `docs/Glowlicense.txt`) in space-physics/ncar-glow, "GLOW MODEL OPEN
    SOURCE ACADEMIC RESEARCH LICENSE AGREEMENT" (UCAR): use, reproduction and
    derivative works "for research, academic, and non-profit purposes"; no
    sale or licensing for a fee of the software "or any work that in any
    manner contains the Software"; copies and derivative works carry the
    agreement and its notice.
  - The `ncarglow` Python packaging: `LICENSE.txt` in the same repo,
    Apache License 2.0.
  - `colortable/glowpython.patch` modifies GLOW's driver; the patched file
    carries a dated notice of the change, as the agreement's section 3b asks.
  - `colortable.json` is GLOW output. Shipping it in a commercial app is an
    open pre-release licence question.
- NOAA SWPC data are public domain.
