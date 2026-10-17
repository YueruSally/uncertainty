# RF-Reward stage-D formal training result

Stage D completed on the server using the frozen 450-instance S0-S8 training
catalog, three independent collection trajectories and a maximum of 5,000
actual CCP100 evaluations per instance. No formal-test instance, OOS result or
formal HV was used.

- Collection tasks: 1,350/1,350.
- Actual CCP100 evaluations: 6,746,828 (maximum 6,750,000).
- Executed-action reward labels: 707,528.
- Fixed RF parameters: 200 trees, depth 8, minimum leaf 5, feature fraction
  0.5; primary fitting unweighted.
- All three fixed models beat the training-fold mean baseline in both
  instance-macro RMSE and MAE. This is predictive evidence only, not a claim
  that RF optimization beats Rule.
- `FORMAL_TRAINING_COMPLETE.json` SHA-256:
  `fd755cbd88448a8fd7dbb73d02e4991806d0eadade28e1fa5b8997d29246a25b`.

| Trajectory | Labels | RF macro RMSE | Baseline macro RMSE | Model SHA-256 | Training-data digest |
|---|---:|---:|---:|---|---|
| 1 | 237,175 | 0.0243030782 | 0.0283596459 | `ef9ccf7ea44b17907af95983e24824cb33d1b9e8a2103b0df46bb5072fd8b323` | `c2af9e19a62f56acf8f5af3d7e16cbeb03a89eef8798ea5e78671e154f1e2bd6` |
| 2 | 235,192 | 0.0234849047 | 0.0275065939 | `00f9a4486f86b0844cc5f19cc546a5598190a15650f66d4fa645477d17a91fc5` | `93f5691735a58438f0dde0edfaa70c44f27fb15ac28ff800467a0f0a6be61174` |
| 3 | 235,161 | 0.0227975441 | 0.0270822566 | `d2fa8f5272af245ece5c288dc531a9f77d850c5b99ede9033205340e5ef97e48` | `0a850a271d01eb813b2d1fd8b5624c82b3518f654cdbbb07bf47b118f5df0942` |

The three server-side model files were independently hashed after completion
and matched the hashes above. The stage-D directory is therefore frozen as the
model source for formal paired blocks 1, 2 and 3 respectively.
