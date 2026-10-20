# E1 figures and descriptor screen

`stable_descriptor_selection.json` records the train/validation-only class-direction screen. Six descriptors pass: `ra_peak`, `ra_std90`, `re_std90`, `rd_peak`, `xy_peak`, and `xy_std90`. The box and scatter figure focuses on the three movement-variability measures (`ra_std90`, `re_std90`, `xy_std90`), which also keep their direction on multilink test. The three peak measures reverse direction on multilink test. Multilink labels do not select model inputs or thresholds. `ra_mean` and `re_mean` do not pass the train/validation screen.

`hierarchical_pca.json` records the training-basis fit counts and explained variance. The one-minute level concatenates six ten-second points; partial coverage is interpolated within the documented limits. `timeseries_ml.pdf` shows a 20-second rolling mean in two rows across train, validation, and test. `mean_maps_movement.pdf` shows consecutive-second map changes from labeled samples.

These visual comparisons are descriptive. Validation and multilink test each contain both classes across separate sessions or captures. Because a capture has one label, the plots cannot by themselves separate occupancy effects from every capture-specific condition.
