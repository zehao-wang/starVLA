# CHANGELOG


## March 30, 2026
1. Fix bugs in QWen3-VL-Fast training pipeline with RoboTwin dataset
    * single node script
    * slurm script (generalizable to any change in node script)
2. Update evaluation pipeline:
    * support general evaluation (random rollout)
    * support evaluation from fixed env init by lerobot - auto detect all gpus for service-client evaluation