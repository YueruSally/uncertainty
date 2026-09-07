# Frozen pilot semantic-audit supplement

This directory was computed from the already frozen pilot CSV and candidate provenance. No optimisation or candidate mutation was run, and the original pilot directory was not modified.

The current executable CCP minimises three separate empirical order-statistic objectives. For objective samples z_1,...,z_S and alpha=0.90, q_alpha = sorted(z)[ceil(alpha*S)-1]. Alpha does not currently define a punctuality feasibility constraint.

`Batches.LT` is a genuine latest-arrival input for every pilot batch. The retained frequency of completion <= LT is reported only as a punctuality diagnostic. It is not CCP feasibility.

The preserved S200 run used population 300, generations 500, S=200, and one pilot run. No full or formal run was launched here.
