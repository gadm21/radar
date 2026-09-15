# E1 — Left/Right Localization (heuristic approach, superseded)

## Task

Classify the occupant's position as **left** vs **right** from mmWave
radar, using the `test_minutes/` recordings that carry a left/right
ground-truth label.

## Approach (abandoned)

A hand-derived physics pipeline: decode raw ADC frames -> range FFT ->
Doppler FFT -> Bartlett beamforming across the 3 RX antennas -> pick the
strongest range-Doppler peak -> map its azimuth to left/right, with a
calibration subset resolving the hardware sign convention and a
confidence-weighted vote aggregating windows into a per-minute call.

## Result

Near-chance discrimination: minute-level balanced accuracy ~0.50 and a
strong bias toward predicting one side regardless of the true label.
With only 3 RX antennas (effective 2-element azimuth aperture at
half-wavelength spacing), weak SNR, and heavy indoor multipath, the
per-frame angle estimate did not reliably separate left from right.

## Superseded by

**E3** — a model-based approach (CNN over range-Doppler + range-azimuth
maps) that learns the spatial signature directly and reaches 0.989
window-level / 1.000 minute-level accuracy under stratified group 5-fold
cross-validation. See `E3/README.md`.
