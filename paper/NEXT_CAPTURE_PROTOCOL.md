# Next capture and evaluation protocol

The existing validation set contains one session per class; the multilink test contains two captures per class. Their class coverage supports evaluation across held-out captures. This optional extension would also test transitions within captures and more fully separate occupancy from session, date, placement, and hardware effects. It should be frozen **before** new captures are collected or viewed.

## Capture plan

1. Record multiple independent sessions at each placement and on different dates. To evaluate transition detection, include both empty and occupied intervals with recorded transitions within these new sessions. Repeat each class and activity at each placement and date. Keep the identity of each recording session in the data table.
2. Log the labeling source and synchronized start/end time for every interval. For occupied intervals, record person count, position relative to radar field of view, stationary versus moving activity, and whether the person is partly or fully outside that field. Preserve natural durations; do not balance the prospective test using its labels.
3. Log device configuration, receiver geometry, radar height/orientation, WiFi transmitter and receiver positions, room dimensions, firmware versions, nominal frame interval, and any changes during a session. Preserve raw packets, radar frames, device timestamps and host timestamps. Record clock offset/drift estimates, packet/frame drops, invalid seconds, and rejection reasons.
4. Include predeclared stress conditions only if they can be measured: controlled WiFi packet loss/interference, stationary occupancy, and positions near or outside the radar field of view. Keep these condition labels so sensitivity and specificity can be reported separately.

## Freeze the analysis before test

1. Assign whole sessions to fit, validation, and untouched test. No original minute, overlapping window, augmentation parent, or mixup partner may cross the fit/validation boundary. Fit PCA, imputation, scaling, feature screens, model weights, early stopping, and any probability orientation on fit or validation only. Save a manifest of all IDs and parent links.
2. Select descriptors, fusion stage, model, decision level, and calibration method using training and grouped validation. Do **not** require a method to pass an accuracy threshold on the prospective test before including it in the reported comparison. Version the exact code, configuration, and per-seed probability archive.
3. For a deployment-style calibration test, set a threshold from an **earlier unlabeled batch** and apply it to later minutes without updating it. Report a frozen validation-selected threshold alongside median and two-means batch calibration. Predefine tie handling, score orientation, and the result for single-class or low-variance batches.
4. Evaluate the untouched sessions once at natural prevalence. Report per-session confusion matrices, sensitivity, specificity, balanced accuracy, AUC, and class counts. Separate single-modality, early fusion, and late fusion; report all fitted fusion weights and every seed. Use paired errors for model comparisons and uncertainty grouped by independent session. Do not treat overlapping windows or adjacent minute buckets as independent replications.

## Acceptance condition

The revised paper can claim occupancy transfer across placement only when test sessions contain both classes, the model and calibration choices are frozen beforehand, and performance remains credible at the session level. Until then, the present results should remain an exploratory analysis of label-associated differences across collection cohorts.
