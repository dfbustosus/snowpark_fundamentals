# NCL Onsite Training — Snowpark ML

Three-day hands-on training using schema isolation (`MLDS_D` database, `MLDS_ROLE_D` role, per-student schemas).

## Prerequisites

- `.env` configured with Snowflake credentials (see root `.env.example`)
- Package installed: `pip install -e ".[dev,notebooks]"` from repo root
- Each student sets `STUDENT_NAME` in their notebook to get an isolated schema

## Schedule

### Day 1 — Experimentation & MLOps

| Time | Notebook | Topics |
|------|----------|--------|
| 1:45–2:30 | `day1/17_experimentation_model_training` | Experiment tracking, feature set comparison, provenance metadata |
| 2:45–3:30 | `day1/18_model_validation_registration` | Holdout validation, champion/challenger, programmatic promotion |
| 3:30–4:00 | `day1/19_deployment_serving_monitoring` | Batch vs real-time inference, drift detection, retraining triggers |

### Day 2 — Advanced Snowpark Patterns

| Time | Notebook | Topics |
|------|----------|--------|
| 2:45–3:30 | `day2/20_advanced_snowpark_patterns` | Scalar & vectorized UDFs, window functions, stored procedures |
| 3:30–4:00 | `day2/21_udfs_realworld_patterns` | Hands-on lab: UDFs, window functions, SP pipeline |

### Day 3 — Capstone Exercises

| Time | Notebook | Topics |
|------|----------|--------|
| 9:15–10:00 | `day3/22_exercise_classification` | End-to-end classification with Feature Store and Registry |
| 11:00–11:45 | `day3/23_exercise_operational` | Retrain, update Feature Store, troubleshoot pipeline |

## Folder Layout

```
onsite/
├── day1/                          # Experimentation, Validation, Deployment
│   ├── 17_*_model_training.ipynb  (+SOLUTION)
│   ├── 18_*_registration.ipynb    (+SOLUTION)
│   └── 19_*_monitoring.ipynb      (+SOLUTION)
├── day2/                          # Advanced Patterns, Hands-On Lab
│   ├── 20_*_patterns.ipynb        (+SOLUTION)
│   └── 21_*_patterns.ipynb        (+SOLUTION)
└── day3/                          # Capstone Exercises
    ├── 22_*_classification.ipynb  (+SOLUTION)
    └── 23_*_operational.ipynb     (+SOLUTION)
```

Each exercise notebook has a corresponding `_SOLUTION` variant for instructor reference.

## Instructor Guides

Located in `docs/` at the repo root:

- `docs/module_17_instructor_guide.md`
- `docs/module_18_instructor_guide.md`
- `docs/module_19_instructor_guide.md`
- `docs/module_20_instructor_guide.md`
- `docs/module_21_instructor_guide.md`
- `docs/module_22_instructor_guide.md`
- `docs/module_23_instructor_guide.md`
- `docs/day3_exercises_instructor_guide.md`
