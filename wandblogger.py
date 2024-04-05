import datetime
import pathlib
import numpy as np
import wandb
import json
import time
from pathlib import Path


class WandbLogger:
    def __init__(self, logdir, config, step):
        self._logdir = Path(logdir)
        assert "/" in str(logdir)
        name = str(logdir).split("/")[-1]
        time_str = datetime.datetime.now().strftime("%m-%d#%H-%M-%S")
        self._logdir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        wandb.init(
            project="pgm-rl",
            dir=str(logdir),
            name=time_str + "_" + name,
            config=config,
            entity="osu-pgm",
        )
        path = pathlib.Path(__file__).parent
        print(f"Log all python codes in {path}")
        wandb.run.log_code(path, exclude_fn=lambda x: ".history" in x)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=None):
        if step is None or step is False:  # Adapted for clarity
            step = self.step
        scalars = self._scalars.copy()
        if fps:
            scalars["fps"] = self._compute_fps(step)
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars.items()))

        # Log scalars
        wandb.log(scalars, step=step)

        # Log images
        for name, value in self._images.items():
            wandb.log({name: wandb.Image(value)}, step=step)

        # Log videos
        for name, value in self._videos.items():
            # Assume value is in (B, T, H, W, C) format; adjust if necessary
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            # WandB expects (T, C, H, W), so reshape accordingly
            value = value.transpose(0, 1, 4, 2, 3)  # Adjust based on your video format
            wandb.log({name: wandb.Video(value[0], fps=4, format="mp4")}, step=step)

        # Optionally, save scalars to a local file as well
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **scalars}) + "\n")

        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time = time.time()  # Update to current time
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        # For logging scalars outside the main logging loop
        wandb.log({"scalars/" + name: value}, step=step)

    def offline_video(self, name, value, step):
        # Handle video logging similarly to the video method
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        # Adjust based on your video format
        value = value.transpose(0, 1, 3, 4, 2)
        wandb.log({name: wandb.Video(value[0], fps=4, format="mp4")}, step=step)
