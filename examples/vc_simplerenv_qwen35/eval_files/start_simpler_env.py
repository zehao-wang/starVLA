import json
import os

from examples.vc_simplerenv_qwen35.eval_files.custom_argparse import get_args
from examples.vc_simplerenv_qwen35.eval_files.model2simpler_interface import ModelClient

import numpy as np


README_PATH = "examples/vc_simplerenv_qwen35/eval_files/README_simplerenv_runtime.md"


def assert_simplerenv_runtime() -> None:
    import cv2

    expected = {
        "numpy": "1.24.0",
        "opencv-python": "4.6.0",
    }
    actual = {
        "numpy": np.__version__,
        "opencv-python": cv2.__version__,
    }
    mismatches = [
        f"{name}: expected {version}, got {actual[name]}"
        for name, version in expected.items()
        if actual[name] != version
    ]
    if mismatches:
        detail = "\n".join(mismatches)
        raise RuntimeError(
            "Unsupported SimplerEnv runtime versions:\n"
            f"{detail}\n"
            f"See {README_PATH} and apply requirements_simplerenv_patch.txt inside simpler_env."
        )


def get_bridge_state_from_env(env) -> np.ndarray:
    """Extract 8D EEF state from a SimplerEnv WidowX environment.

    State layout (matches bridge_rt_1 dataset modality.json):
        [eef_x, eef_y, eef_z, roll, pitch, yaw, pad(0), gripper_open]

    - EEF pose is expressed in the robot base frame.
    - SAPIEN Pose.q is [w, x, y, z]; transforms3d.euler.quat2euler expects [w, x, y, z].
    - Gripper is normalised from joint qpos [0.015, 0.037] → [0.0, 1.0].
    """
    from transforms3d.euler import quat2euler

    tcp_in_base = env.agent.robot.pose.inv() * env.tcp.pose
    eef_pos = tcp_in_base.p  # [x, y, z] in base frame

    # SAPIEN quaternion is [w, x, y, z]
    roll, pitch, yaw = quat2euler(tcp_in_base.q, axes="sxyz")

    # Gripper joint qpos in [0.015, 0.037] → normalised to [0, 1]
    qpos = env.agent.robot.get_qpos()
    gripper_open = float(np.clip((qpos[-2] - 0.015) / (0.037 - 0.015), 0.0, 1.0))

    return np.array(
        [eef_pos[0], eef_pos[1], eef_pos[2], roll, pitch, yaw, 0.0, gripper_open],
        dtype=np.float32,
    )


def run_starvla_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    """Episode loop that always passes robot EEF state to model.step()."""
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from simpler_env.utils.visualization import write_video
    from transforms3d.euler import quat2euler

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(env_name, **additional_env_build_kwargs, **kwargs)

    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {"init_xy": np.array([obj_init_x, obj_init_y])}
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {"episode_id": obj_episode_id}

    obs, _ = env.reset(options=env_reset_options)
    is_final_subtask = env.is_final_subtask()

    task_description = instruction if instruction is not None else env.get_language_instruction()
    print(task_description)

    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    model.reset(task_description)

    timestep = 0
    success = "failure"

    while not (predicted_terminated or truncated):
        robot_state = get_bridge_state_from_env(env)
        raw_action, action = model.step(image, task_description, robot_state=robot_state)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated and not is_final_subtask:
            predicted_terminated = False
            env.advance_to_next_subtask()

        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # ---- save video ----
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path.rstrip("/").split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    else:
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    rgb_overlay_str = (
        os.path.splitext(os.path.basename(rgb_overlay_path))[0]
        if rgb_overlay_path is not None
        else "None"
    )
    r, p, y = quat2euler(robot_init_quat)
    video_path = (
        f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/"
        f"rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_str}/{video_name}"
    )
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def starvla_evaluator(model, args):
    from simpler_env.utils.env.env_builder import get_robot_control_mode

    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_starvla_eval_single_episode(
                                    obj_init_x=obj_init_x, obj_init_y=obj_init_y, **kwargs
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(
                            run_starvla_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs)
                        )
                else:
                    raise NotImplementedError()

    return success_arr


def write_task_summary(success_arr, args):
    """Write a per-task JSON summary to logging_dir."""
    n_total = len(success_arr)
    n_success = int(sum(success_arr))
    summary = {
        "task": args.env_name,
        "ckpt_path": args.ckpt_path,
        "total_episodes": n_total,
        "successes": n_success,
        "success_rate": round(n_success / n_total, 4) if n_total > 0 else 0.0,
        "per_episode": [bool(s) for s in success_arr],
    }
    os.makedirs(args.logging_dir, exist_ok=True)
    out_path = os.path.join(args.logging_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_path}")
    return out_path


if __name__ == "__main__":
    assert_simplerenv_runtime()
    args = get_args()

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        host=args.host,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,
    )

    success_arr = starvla_evaluator(model, args)
    write_task_summary(success_arr, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
