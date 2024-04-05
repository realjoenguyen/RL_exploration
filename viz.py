import argparse
from pathlib import Path
import sys
import os
from typing import List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from termcolor import cprint
from tqdm import tqdm

PointsType = List[Tuple[float, float, int]]  # x, y, episode_id


from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(points: PointsType, first_episodes, output_path):
    # Separate x, y, and episode_id values
    x, y, episode_ids = zip(*points)

    # Normalize episode_id values for color mapping
    max_episode_id = max(episode_ids)
    print(f"Draw color for episode id from 1 to {max_episode_id}")
    # colors = [episode_id / max_episode_id for episode_id in episode_ids]
    #
    fig, ax = plt.subplots()
    # Adjust scatter to map colors with vmin and vmax for correct colorbar scale
    scatter = ax.scatter(
        x, y, c=episode_ids, cmap="viridis", vmin=1, vmax=max_episode_id
    )

    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Add colorbar with label
    plt.colorbar(scatter, ax=ax, label="Episode ID")

    # caption for the plot
    plt.title(f"Scatter plot of the first {first_episodes} episodes")

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()  # Shows the plot
    # save the plot
    save_path = Path(output_path) / f"scatter_plot_{first_episodes}.png"
    cprint(f"saving scatter plot to {save_path}", "green")
    # check if save_path.parent exists
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)

    # plt.close(fig) is commented out to display the plot when running interactively
    return image


if __name__ == "__main__":
    # # logdir = sys.argv[1]
    # logdir = "./logdir/gymnasiummaze_PointMaze_UMazeDense-v3-MCD-expl_and_task-seed0/train_eps"
    # FIRST_EPISODES = 100
    # if not os.path.isdir(logdir):
    #     print("Error: {} is not a directory".format(logdir))
    #     sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Visualize the first num_episodes episodes in logdir"
    )
    parser.add_argument("--logdir", type=str, help="Directory containing the npz files")
    parser.add_argument(
        "--num_episodes", nargs="+", help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--output", type=str, default=".", help="Directory containing the npz files"
    )

    args = parser.parse_args()
    episode_marks = (
        args.num_episodes
        if isinstance(args.num_episodes, list)
        else [args.num_episodes]
    )
    episode_marks = [int(e) for e in episode_marks]
    print("Print episode_marks", episode_marks)

    logdir = args.logdir

    # take all npz files in the logdir
    # sorted by the first timestep in format: timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),         self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
    files = [
        f for f in os.listdir(logdir) if f.endswith(".npz") and f.startswith("2024")
    ]
    assert len(files) > 0, "No npz files found in {}".format(logdir)
    files.sort(key=lambda f: f.split("-")[0])

    # each file contains a dictionary with the following
    # 'state', 'action', 'reward', etc.
    # record the first 2 number in each state - the x and y position and the episode id number - indicated by 'is_first' in the dictionary

    episode_id = 0
    points = []
    # done_collect_points = False
    print(f"Found {len(files)} eps files")

    # name_exp = logdir.split("/")[-2]
    mark_id = 0
    done = False
    for f in tqdm(files):
        f_path = os.path.join(logdir, f)
        data = np.load(f_path)
        states = data["state"]
        for i in range(len(states)):
            # print(f"{states[i][0]}, {states[i][1]}, {episode_id}")
            if data["is_first"][i]:
                # print("eps_id=", episode_id)
                episode_id += 1
                need_num_episode = episode_marks[mark_id]

                if episode_id > need_num_episode:
                    done_collect_points = True
                    print(f"Done collecting {need_num_episode} episodes.")
                    scatter_plot(points, need_num_episode, args.output)
                    mark_id += 1
                    if mark_id == len(episode_marks):
                        print("Done all episode marks")
                        done = True
                        break

            points.append((states[i][0], states[i][1], episode_id))
            if done:
                break

        # if done_collect_points:
        #     break


# # print(f"Draw scatter plot for {len(points)}")
# scatter_plot(points, FIRST_EPISODES, args.output)
