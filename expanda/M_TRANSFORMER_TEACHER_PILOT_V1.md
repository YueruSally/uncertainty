# M Transformer Teacher Pilot V1

## Purpose

This pilot is the first mutation experiment in which a Transformer actually
selects the replacement batch and path.  The previously validated
`REPLACE_G10_JOINT_BALANCED` rule is used only as a teacher.

## Controlled comparison

All three methods use the same frozen CCP scenarios (`S=200`, `alpha=0.90`),
population 20 and 400 real objective evaluations.

- `BASE`: original mutation.
- `TEACHER_BALANCED`: force Replace from generation 10 and use the deterministic
  Balanced selector.
- `TRANSFORMER_BC`: use the same generation-10 Replace schedule, random stream,
  execution and repair as the teacher, but let the path-v4 Transformer select
  the joint batch/path candidate.

The only Teacher/Transformer difference is the candidate selector.

## Pipeline

1. Collect late-Replace states on seven instances and two seeds.
2. Recompute the deterministic Balanced action for every saved state without
   additional CCP objective evaluations.
3. Train the real path-v4 M Transformer by behaviour cloning.  `S0_011` is a
   validation instance and is excluded from gradient updates.
4. Evaluate Base, Teacher and Transformer on three unseen instances
   (`S0_010`, `S0_012`, `S0_013`) and three new seeds.

Run `run_m_balanced_teacher_transformer_pilot_v1.sh`.  The final compact result
is written to `QUICK_RESULT.json` under the pilot output root.

## Admission rule

The pilot is promising only when all of the following hold:

- held-out teacher imitation top-1 accuracy is at least 50%;
- Transformer mean paired q90-HV improvement over Base is positive;
- Transformer wins at least 5 of the 9 matched runs.

Passing this pilot authorises a larger held-out validation and subsequent PPO
fine-tuning.  It is not itself the final paper result.
