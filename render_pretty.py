import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import cv2
import numpy as np
import yaml
from gymnasium.envs.registration import register
from matplotlib import pyplot as plt
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

register(
    id="CustomDoublePendulum-v0",
    entry_point="inverted_double_pendulum:InvertedDoublePendulumEnv",
)

switch_order = [3, 1, 2, 0, 1, 2]


CAMERA_READY_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5.0,
    # Move the lookat a bit if you'd like the camera to center higher/lower
    "lookat": np.array([0.0, 0.0, 0.2]),
    # Add these two to get a 3D angle
    "azimuth": 90.0,
    "elevation": -20,
}

FAST_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5.0,
    # Move the lookat a bit if you'd like the camera to center higher/lower
    "lookat": np.array([0.0, 0.0, 0.2]),
    # Add these two to get a 3D angle
    "azimuth": 90.0,
    "elevation": -20,
}


def make_white_background_transparent(bgr_image, threshold=250):
    """
    Convert BGR image to BGRA, treating all near-white pixels as fully transparent.
    threshold: how close to white (255,255,255) must a pixel be to be considered background.
    """
    # Convert to BGRA
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)

    # Create a mask where pixels are "white" above a certain threshold
    white_mask = cv2.inRange(
        bgr_image, (threshold, threshold, threshold), (255, 255, 255)
    )
    # Set alpha=0 for masked (white) pixels
    bgra[white_mask > 0, 3] = 0

    return bgra


def render_viz(visualization, save_dir, button_image_path):
    """
    All the logic you had inside the per-visualization loop:
    - mkdir for vis_dir
    - load model
    - set up vec_env
    - start ffmpeg procs
    - step/render loop
    - close procs
    - plotting and stats dumping
    """
    name = visualization["name"]
    console = Console()
    console.print(f"[bold green]Rendering visualization:[/bold green] {name}")

    vis_dir = Path(save_dir) / name
    vis_dir.mkdir(parents=True, exist_ok=True)

    # — load & configure env/model exactly as before —
    # — launch ffmpeg subprocesses —
    # — run the step/render loop, write to stdin of each ffmpeg proc —
    # — wait() on each proc —
    # — do your matplotlib saving and stats file —

    name = visualization.get("name")

    vis_dir = save_dir / name

    # Make the visualization directory if it doesn't exist
    vis_dir.mkdir(parents=True, exist_ok=True)

    if visualization["recording"]["fast"]:
        # Set resolution to 640x480
        resolution = (640, 480)
        # Set the camera configuration to fast
        camera_config = FAST_CAMERA_CONFIG
    else:
        resolution = (3840, 2160)  # 4K resolution
        camera_config = CAMERA_READY_CAMERA_CONFIG

    # Whether or not to render transparent background
    transparent = visualization["recording"]["alpha"]

    # Load model
    model_path = visualization["model"]["path"]
    if not model_path == "":
        model = PPO.load(model_path)
    else:
        model = None

    mode_switch_steps = visualization["env"]["mode_switch_steps"]
    switch_order = visualization["env"]["switch_order"]

    init_qpos = None
    if "init_qpos" in visualization["env"]:
        init_qpos = np.array(visualization["env"]["init_qpos"])

    vec_env = make_vec_env(
        "CustomDoublePendulum-v0",
        n_envs=1,
        env_kwargs={
            "mode_switch_steps": visualization["env"]["mode_switch_steps"],
            "switch_order": visualization["env"]["switch_order"],
            "camera_config": camera_config,
            "render_resolution": resolution,
            "terminate_on_dead": False,
            "init_qpos": init_qpos,
        },
    )

    if not visualization["recording"]["fast"]:

        frame_skip = vec_env.envs[0].unwrapped.frame_skip

        # Set frame skip to 1
        vec_env.envs[0].unwrapped.frame_skip = 1
        # Update render fps to match the frame skip
        vec_env.envs[0].unwrapped.metadata["render_fps"] = 1.0 / (
            vec_env.envs[0].unwrapped.dt
        )

        vec_env.envs[0].unwrapped.mode_switch_steps *= frame_skip
    else:

        frame_skip = 1

    obs = vec_env.reset()

    # Test render once to get frame size
    bgr_frame = vec_env.render()  # Typically BGR with shape [H, W, 3]

    if transparent:
        bgra_frame = make_white_background_transparent(bgr_frame)
    else:
        bgra_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)

    # We can access the viewer in case we want to change camera angles
    # viewer = vec_env.envs[0].unwrapped.mujoco_renderer.viewer
    # viewer.cam.azimuth = 110
    # viewer.cam.elevation = -30

    fps = vec_env.envs[0].unwrapped.metadata["render_fps"]

    video_path_main = vis_dir / f"{name}_video.mov"

    # --------------------
    # 1) Start an FFmpeg subprocess in ProRes 4444 mode
    # --------------------
    ffmpeg_cmd_main = [
        "ffmpeg",
        "-y",  # overwrite output file if it exists
        "-loglevel",
        "error",  # suppress all but error messages
        "-f",
        "rawvideo",  # input format is raw video data
        "-pix_fmt",
        "bgra",  # our frames are in BGRA channel order
        "-s",
        f"{resolution[0]}x{resolution[1]}",
        "-r",
        str(fps),
        "-i",
        "-",  # read from stdin
        "-c:v",
        "prores_ks",  # use the prores_ks encoder
        "-pix_fmt",
        "yuva444p10le",  # 10-bit YUV + alpha
        video_path_main,
    ]

    video_path_left_button = vis_dir / f"{name}_left_button.mov"

    # Open the button image and get its reoslution
    button_img = cv2.imread(button_image_path, cv2.IMREAD_UNCHANGED)

    button_resolution = button_img.shape

    ffmpeg_cmd_left_button = [
        "ffmpeg",
        "-y",  # overwrite output file if it exists
        "-loglevel",
        "error",  # suppress all but error messages
        "-f",
        "rawvideo",  # input format is raw video data
        "-pix_fmt",
        "bgra",  # our frames are in BGRA channel order
        "-s",
        f"{button_resolution[0]}x{button_resolution[1]}",
        "-r",
        str(fps),
        "-i",
        "-",  # read from stdin
        "-c:v",
        "prores_ks",  # use the prores_ks encoder
        "-pix_fmt",
        "yuva444p10le",  # 10-bit YUV + alpha
        video_path_left_button,
    ]

    video_path_right_button = vis_dir / f"{name}_right_button.mov"
    ffmpeg_cmd_right_button = [
        "ffmpeg",
        "-y",  # overwrite output file if it exists
        "-loglevel",
        "error",  # suppress all but error messages
        "-f",
        "rawvideo",  # input format is raw video data
        "-pix_fmt",
        "bgra",  # our frames are in BGRA channel order
        "-s",
        f"{button_resolution[0]}x{button_resolution[1]}",
        "-r",
        str(fps),
        "-i",
        "-",  # read from stdin
        "-c:v",
        "prores_ks",  # use the prores_ks encoder
        "-pix_fmt",
        "yuva444p10le",  # 10-bit YUV + alpha
        video_path_right_button,
    ]

    ffmpeg_proc_main = subprocess.Popen(ffmpeg_cmd_main, stdin=subprocess.PIPE)
    ffmpeg_proc_left_button = subprocess.Popen(
        ffmpeg_cmd_left_button, stdin=subprocess.PIPE
    )
    ffmpeg_proc_right_button = subprocess.Popen(
        ffmpeg_cmd_right_button, stdin=subprocess.PIPE
    )

    # Print the details of the visualization, tabbed in 1
    console.print(
        f"[green]Ouptut path: [bold]{vis_dir}[/bold][/green]\n"
        f"[green]FPS: [bold]{fps}[/bold][/green]\n"
        f"[green]Resolution: [bold]{resolution}[/bold][/green]\n"
        f"[green]Alpha: [bold]{transparent}[/bold][/green]\n"
        f"[green]Model: [bold]{model_path}[/bold][/green]\n"
    )

    # (Optional) If you also want a live display:
    cv2.namedWindow("Pendulum", cv2.WINDOW_NORMAL)

    ignore_done = visualization["env"]["ignore_done"]
    deterministic = visualization["model"]["deterministic"]

    total_steps = mode_switch_steps * len(switch_order)

    # Prepare containers
    actions_list = []
    rewards_list = []
    distance_penalty_list = []
    velocity_penalty_list = []
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("Mode: [magenta]{task.fields[target_mode]}", justify="right"),
            transient=True,  # auto‑clear when done
        ) as progress:

            task = progress.add_task("Rendering", total=total_steps, target_mode=0)

            for step in range(total_steps):  # or however many steps you want
                if model is not None:
                    action, _states = model.predict(obs, deterministic=deterministic)
                else:
                    # Zero out the action
                    action = np.zeros(vec_env.action_space.shape)
                    action = np.expand_dims(action, axis=0)

                # Manually skip frames, but record data for each video frame
                for _ in range(frame_skip):

                    obs, rewards, dones, info = vec_env.step(action)

                    target_mode = info[0]["target_mode"]

                    # (Optional) Change camera angles or other viewer settings
                    # viewer.cam.azimuth += 1

                    # Render the environment (BGR by default for MuJoCo/gym)
                    bgr_frame = vec_env.render()
                    if bgr_frame is None:
                        # Some envs return None if they can't render
                        continue

                    # 2) Convert background to transparent
                    if transparent:
                        bgra_frame = make_white_background_transparent(
                            bgr_frame, threshold=250
                        )
                    else:
                        bgra_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)

                    # 3) Send the BGRA data to ffmpeg
                    #    We write raw bytes row-by-row in BGRA format
                    ffmpeg_proc_main.stdin.write(bgra_frame.tobytes())

                    # (Optional) Show or debug in a local window
                    # Remember OpenCV doesn't display alpha, so it will appear black behind
                    display_bgr = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)
                    cv2.imshow("Pendulum", display_bgr)

                    # Example break condition
                    if dones[0] and not ignore_done:
                        print("RESET")
                        obs = vec_env.reset()

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    # 3) Send the button image to ffmpeg
                    if action[0][0] < -0.05:
                        # left button pressed
                        # 3) Send the BGRA data of the left button image to ffmpeg
                        ffmpeg_proc_left_button.stdin.write(button_img.tobytes())

                    else:
                        # Left button not pressed
                        # Send black image to ffmpeg
                        black_img = np.zeros(
                            (button_resolution[0], button_resolution[1], 4),
                            dtype=np.uint8,
                        )
                        ffmpeg_proc_left_button.stdin.write(black_img.tobytes())

                    if action[0][0] > 0.05:
                        # right button pressed
                        # 3) Send the BGRA data of the right button image to ffmpeg
                        ffmpeg_proc_right_button.stdin.write(button_img.tobytes())
                    else:
                        # Right button not pressed
                        # Send black image to ffmpeg
                        black_img = np.zeros(
                            (button_resolution[0], button_resolution[1], 4),
                            dtype=np.uint8,
                        )
                        ffmpeg_proc_right_button.stdin.write(black_img.tobytes())

                # record for later plotting
                # If action is an array, we copy it; otherwise record the scalar
                actions_list.append(np.array(action).copy())
                # rewards is typically a 1‑element array or list
                rewards_list.append(float(rewards[0]))

                distance_penalty_list.append(float(info[0]["distance_penalty"]))
                velocity_penalty_list.append(float(info[0]["velocity_penalty"]))

                # Update the progress bar
                progress.update(task, advance=1, target_mode=target_mode)

        # ----------------------------------
        # Now: plot Actions vs Step
        # ----------------------------------
        output_path = vis_dir / f"{name}_actions.svg"
        actions_arr = np.stack(actions_list)  # shape (N, action_dim) or (N,)

        fig, ax = plt.subplots()

        # plot your data (no markers, just a line)
        if actions_arr.ndim == 1:
            ax.plot(actions_arr, linewidth=1)
        else:
            for i in range(actions_arr.shape[1]):
                ax.plot(actions_arr[:, i], linewidth=1)

        # 1) turn _everything_ off
        ax.set_ylim(-1, 1)
        ax.set_axis_off()

        # 2) save with no padding and a transparent background
        fig.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",  # crop to your line’s extents
            pad_inches=0,  # no extra margin
            transparent=True,  # so only the path shows
        )
        plt.close(fig)

        # ----------------------------------
        # Then: plot Rewards vs Step
        # ----------------------------------
        output_path = vis_dir / f"{name}_rewards.svg"
        rewards_arr = np.array(rewards_list)

        fig, ax = plt.subplots()
        ax.plot(rewards_arr, linewidth=1)
        ax.set_ylim(-8, 10)
        ax.set_axis_off()

        fig.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig)

        # Plot the reward components
        # Scale and shift both reward components to -10 to 10
        # so we can plot them together
        distance_penalty_list = np.array(distance_penalty_list)
        velocity_penalty_list = np.array(velocity_penalty_list) * 1e4

        output_path = vis_dir / f"{name}_reward_components.svg"
        fig, ax = plt.subplots()
        ax.plot(distance_penalty_list, linewidth=1, label="Distance penalty")
        ax.plot(velocity_penalty_list, linewidth=1, label="Velocity penalty")
        ax.set_ylim(-20, 10)
        ax.set_axis_off()
        ax.legend()
        fig.savefig(
            output_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
        plt.close(fig)

        total_reward = np.sum(rewards_arr)

        # Write statistics to a text file
        stats_path = vis_dir / f"{name}_stats.txt"
        with open(stats_path, "w") as f:
            f.write(f"Total reward: {total_reward}\n")
            f.write(f"Total steps: {total_steps}\n")
            f.write(f"Average reward: {total_reward / total_steps}\n")
    except KeyboardInterrupt:
        Console().print(f"[yellow]Interrupted rendering {name}, cleaning up…[/yellow]")

    finally:
        # --------------------
        # 4) Cleanly close FFmpeg
        # --------------------
        ffmpeg_proc_main.stdin.close()
        ffmpeg_proc_main.wait()

        ffmpeg_proc_left_button.stdin.close()
        ffmpeg_proc_left_button.wait()

        ffmpeg_proc_right_button.stdin.close()
        ffmpeg_proc_right_button.wait()

        cv2.destroyAllWindows()


@click.command()
@click.option("--save-dir", default="output", help="Where to save outputs.")
@click.option(
    "--workers",
    "-w",
    default=1,
    type=int,
    help="Number of parallel workers for rendering.",
)
@click.argument("config", type=click.Path(exists=True))
def main(save_dir, workers, config):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    if "visualizations" not in cfg:
        raise click.UsageError("config.yml must have top-level 'visualizations'")
    console = Console()

    # Check if config has a save_dir field
    if "save_dir" in cfg:
        save_dir = Path(cfg["save_dir"])

        # Warn user, in yellow, that the save_dir is being overridden
        console.print(
            f"[bold yellow]Warning: Save directory overridden to [bold]{save_dir}[/bold][/bold yellow]"
        )

    console.print(
        "[bold underline green]Visualizations to be rendered:[/bold underline green]"
    )
    for viz in cfg["visualizations"]:
        console.print(f"  • [yellow]{viz.get('name')}[/yellow]")

    # locate button image once
    button_image_path = Path(
        cfg.get("button_image_path", "assets/button_placeholder.png")
    )
    if not button_image_path.exists():
        raise click.FileError(str(button_image_path), hint="button image not found")

    viz_list = cfg["visualizations"]

    if workers > 1:
        console.print(f"[cyan]Spawning up to {workers} workers…[/cyan]")
        exe = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {
                exe.submit(render_viz, viz, save_dir, button_image_path): viz
                for viz in viz_list
            }
            for fut in as_completed(futures):
                viz = futures[fut]
                # re-raise any exceptions from child
                fut.result()
        except KeyboardInterrupt:
            console.print(
                "[yellow]Interrupted by user, shutting down executor…[/yellow]"
            )
            exe.shutdown(cancel_futures=True)
            sys.exit(1)
        finally:
            exe.shutdown(wait=False, cancel_futures=True)
    else:
        try:
            for viz in viz_list:
                render_viz(viz, save_dir, button_image_path)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted by user[/yellow]")
            sys.exit(1)


if __name__ == "__main__":
    main()
