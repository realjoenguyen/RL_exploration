We investigate three exploration methods in model-based RL setting: 
- curiosity
- Disagreement (Plan2Explore)
- Monte Carlo dropout

## Wandb reports
https://wandb.ai/osu-pgm/pgm-rl

## Presentation:
please find at the presentation file :) 

## Instructions
Add `MUJOCO_GL=egl` before python command if you have this error

```
self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.FatalError: gladLoadGL error
```


Get dependencies with python 3.9:
```
pip install -r requirements.txt
```
Run training on Point Maze task:
```
python3 dreamer.py --configs gymnasium_maze --task gymnasiummaze_PointMaze_UMazeDense-v3 --logdir ./logdir/gymnasiummaze_PointMaze_UMazeDense-v3
```

Run Plan2Explore:
```
python3 dreamer.py --configs gymnasium_maze_p2e --task gymnasiummaze_PointMaze_UMazeDense-v3 --logdir ./logdir/gymnasiummaze_PointMaze_UMazeDense-v3-P2E-seed0 --seed 0
```

Run MC-Dropout (ONLY EXPLORATION):
```
python3 dreamer.py --configs gymnasium_maze_mcd --task gymnasiummaze_PointMaze_UMazeDense-v3 --logdir ./logdir/gymnasiummaze_PointMaze_UMazeDense-v3-MCD-seed0 --seed 0
```

Run MC-Dropout (EXPLORATION (2e5) and TASK for remaining 3e5):
```
python3 dreamer.py --configs gymnasium_maze_mcd_expl_and_task --task gymnasiummaze_PointMaze_UMazeDense-v3 --logdir ./logdir/gymnasiummaze_PointMaze_UMazeDense-v3-MCD-expl_and_task-seed0 --seed 0
```

Monitor results:
```
tensorboard --logdir ./logdir --port 1111 --bind_all
```

## Acknowledgments
This code is heavily adapted from:
- https://github.com/NM512/dreamerv3-torch
