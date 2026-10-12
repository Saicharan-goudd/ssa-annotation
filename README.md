# ssa-annotation
Pipeline to annotate stars and streaks in SSA FITS images: pre-processing, 1024x1024 tiling, and YOLO segmentation label export.
## Overview
This pipeline annotates two feature classes in single-band 16-bit FITS
images of the sky (9568 x 6380 px):

- Class 0, star: compact blobs
- Class 1, streak: elongated tracks of satellites or debris

## Method
1. Pre-processing: block-median background subtraction, MAD noise
   estimate, and an asinh stretch to 8-bit.
2. Detection: Gaussian-smoothed hysteresis thresholding (seed at 5 sigma,
   grown to 2.5 sigma), minimum area 4 px.
3. Classification: shape-based, using elongation, length and fill ratio.
   Borderline cases are flagged for manual review in catalog.csv.
4. Tiling: images are padded to a multiple of 1024 and cut into exact
   1024 x 1024 tiles, so no data is lost. Tiles are repatched and checked
   against the original.
5. Export: YOLO Ultralytics segmentation labels, one file per tile.

## Assumptions
- Noise is roughly uniform across the frame.
- Stars are near-circular; streaks are clearly elongated.
- Isolated 1-3 px peaks are hot pixels or cosmic rays and are not labelled.
