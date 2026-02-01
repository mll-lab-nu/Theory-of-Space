#!/usr/bin/env python3
"""
Split SpatialGym runner: separate exploration, evaluation, and cogmap phases.

Phases:
- exploration: Run dataset creation only (generates exploration histories)
- evaluation: Build eval messages and run inference
- cogmap: Build cogmap messages and run inference
"""
import argparse
import os
import sys
import shlex
import subprocess
import time
import socket
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml as pyyaml
import urllib.request
import threading
from datetime import datetime
from vagen.env.spatial.llm_inference import run_inference_for_combo_dirs, reevaluate_combo_dirs, reevaluate_cogmaps_combo_dirs, reevaluate_cogmap_fb_combo_dirs
from vagen.env.spatial.Base.tos_base.utils.env_logger import SpatialEnvLogger
from vagen.env.spatial.Base.tos_base.utils import  get_model_name
from vagen.env.spatial.common import STATE_BASENAME
SCRIPT_DIR = Path(__file__).resolve().parent


def is_port_available(port: int, host: str = '0.0.0.0') -> bool:
    """Check if a port is available (not in use)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except (socket.error, OSError):
        return True


def find_available_port(start_port: int = 5000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")


def get_adaptive_port(user_port: int = None, default_start: int = 5000) -> int:
    """Get a port for the server. If user_port is specified and available, use it.
    Otherwise, find an available port starting from default_start."""
    if user_port is not None:
        if is_port_available(user_port):
            return user_port
        else:
            print(f"Warning: User-specified port {user_port} is not available, finding alternative...")
            return find_available_port(default_start)
    return find_available_port(default_start)


def parse_args():
    p = argparse.ArgumentParser(
        description="SpatialGym runner with separated phases: exploration, evaluation, cogmap."
    )
    # Phase selection
    p.add_argument("--phase", type=str, default="all",
                   choices=['explore', 'explore_fb', 'eval', 'cogmap', 'cogmap_fb', 'all', 'aggregate', 'reeval', 'cogmap_reeval', 'cogmap_fb_reeval'],
                   help="Which phase to run: explore, explore_fb, eval, cogmap, cogmap_fb, reeval, cogmap_reeval, cogmap_fb_reeval, aggregate, all")

    # Common parameters
    p.add_argument("--exp-type", type=str, dest="exp_type", 
                   default="active",
                   help="Experiment type: active, passive, or comma-separated for multiple (e.g., 'active,passive'). Default: active")
    p.add_argument("--model-name", type=str, default="gpt-4o-mini",
                   help="Model identifier. Default: gpt-4o-mini")
    p.add_argument("--data-dir", type=str, dest="data_dir", default=None, 
                   help="Data directory root. Default: data")
    p.add_argument("--output-root", type=str, dest="output_root", default="results", 
                   help="Root dir for output. Default: results")
    p.add_argument("--num", type=int, default=1, 
                   help="Number of samples per task (exploration phase). Default: 1")
    p.add_argument("--render-mode", type=str, dest="render_mode", default="text", 
                   help="Environment render mode: vision, text, or comma-separated for multiple (e.g., 'vision,text'). Default: text")
    p.add_argument("--max-exp-steps", type=int, dest="max_exp_steps", default=20,
                   help="Max steps for exploration. Default: 20")
    p.add_argument("--seed-range", type=str, dest="seed_range", default=None, 
                   help="Seed range 'start-end' (0-based), e.g., 0-24")
    p.add_argument("--enable-think", type=int, dest="enable_think", choices=[0,1], default=1, 
                   help="1 to enable think, 0 to disable (default: 1)")
    # p.add_argument("--proxy-agent", type=str, dest="proxy_agent", default="strategist", 
    #                choices=["scout","strategist","oracle"], 
    #                help="Proxy agent for passive tasks (required if exp-type is passive)")
    p.add_argument("--all-override", action="store_true", dest="all_override", 
                   help="Override all history (delete whole sample path)")
    p.add_argument("--replay", action="store_true", dest="replay",
                   help="Replay mode: override existing turn logs instead of appending")
    
    # Evaluation/Cogmap phase parameters
    p.add_argument("--eval-task-counts", type=str, dest="eval_task_counts", default=None,
                   help='JSON string for eval task counts, e.g., {"dir": 1}. If omitted, use inference_config.yaml eval_task_counts')
    p.add_argument("--tasks", nargs='+', type=str, dest="tasks", default=None,
                   help="List of tasks to run (e.g. 'dir' 'pov'). If not provided, run all tasks in eval-task-counts.")
    p.add_argument("--eval-override", action="store_true", dest="eval_override",
                   help="Override evaluation history (delete evaluation json only)")
    p.add_argument("--eval-mode", type=str, dest="eval_mode", default="default",
                   choices=["default", "prompt_cogmap", "use_gt_cogmap", "use_model_cogmap"],
                   help="Evaluation mode for cogmap handling: default (no cogmap), prompt_cogmap (ask model to output cogmap first), use_gt_cogmap (provide ground truth cogmap), use_model_cogmap (provide model's last global cogmap). Default: default")
    p.add_argument("--cogmap-override", action="store_true", dest="cogmap_override",
                   help="Override cognitive map cache (regenerate cogmap prompts)")
    p.add_argument("--cogmap-last-global-only", action="store_true", dest="cogmap_last_global_only",
                   help="Run only the last global cogmap per sample")
    p.add_argument("--false-belief-override", action="store_true", dest="false_belief_override",
                   help="Override false belief experiment cache (delete false belief json)")
    p.add_argument("--cogmap-fb-override", action="store_true", dest="cogmap_fb_override",
                   help="Override false belief cognitive map cache (regenerate false belief cogmap prompts)")
    
    # Inference parameters
    p.add_argument("--inference-mode", type=str, dest="inference_mode", 
                   choices=['batch', 'direct'], default='direct',
                   help="Inference mode: batch (OpenAI/Gemini batch API) or direct. Default: direct")
    
    # Server options
    p.add_argument("--no-server", action="store_true", dest="no_server", 
                   help="Do not start internal env server (assume external server is running)")
    p.add_argument("--server-host", type=str, dest="server_host", default="127.0.0.1", 
                   help="Server host to bind/connect")
    p.add_argument("--server-port", type=int, dest="server_port", default=5000, 
                   help="Server port to bind/connect")
    
    # Base config paths
    p.add_argument("--base-env", type=str, dest="base_env", 
                   default=str(SCRIPT_DIR / "base_env_config.yaml"))
    p.add_argument("--base-infer", type=str, dest="base_infer", 
                   default=str(SCRIPT_DIR / "inference_config.yaml"))
    p.add_argument("--base-model", type=str, dest="base_model", 
                   default=str(SCRIPT_DIR / "base_model_config.yaml"))
    
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return pyyaml.safe_load(f)


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        pyyaml.safe_dump(data, f, sort_keys=False)

def build_tmp_paths(run_id: str, task_key: str) -> Dict[str, Path]:
    base = SCRIPT_DIR / "tmp" / run_id / task_key
    return {
        "base": base,
        "env": base / "env.yaml",
        "infer": base / "inference.yaml",
        "model": base / "model.yaml",
    }


def patch_env_yaml(exp_type: str, render_mode="vision",
                   seed_opts: tuple[int, int] | None = None, enable_think: int | None = None,
                   data_dir: str | None = None, proxy_agent: str | None = None,
                   room_config: Dict[str, Any] | None = None, false_belief_exp: bool = False, 
                   max_exp_steps: int = 20, replay: bool = False) -> Dict[str, Any]:
    """Build env config directly without relying on custom_envs.

    Args:
        env_cfg: Base environment config (not used, kept for compatibility)
        exp_type: 'active' or 'passive'
        num: Number of samples
        render_mode: Render mode
        seed_opts: Seed range tuple
        enable_think: Enable thinking
        data_dir: Data directory
        proxy_agent: Proxy agent for passive mode
        room_config: Room configuration (n_objects, room_num, topology, room_size)

    Returns:
        Dict with task config
    """

    # Use a generic task key
    task_key = "SpatialTask"

    # Build environment config directly
    env_config = {
        'exp_type': exp_type,
        'max_exp_steps': 1 if exp_type == 'passive' else max_exp_steps,
        'render_mode': render_mode,
        'prompt_config': {},
        'false_belief_exp': false_belief_exp,
        'replay': replay,
    }

    # Add optional configurations
    if data_dir:
        env_config["data_dir"] = data_dir

    env_config.setdefault("kwargs", {})
    env_config["kwargs"]["seed_start"] = int(seed_opts[0])
    env_config["kwargs"]["seed_end"] = int(seed_opts[1])

    if enable_think is not None:
        env_config["prompt_config"]["enable_think"] = bool(enable_think)

    if proxy_agent and exp_type == "passive":
        env_config["proxy_agent"] = proxy_agent

    # Add room_config if provided
    if room_config:
        env_config["room_config"] = room_config

    # Build the complete task config
    selected = {
        "env_name": "spatial",
        "env_config": env_config,
        "train_size": 1,
        "test_size": int(seed_opts[1] - seed_opts[0] + 1)
    }

    return {task_key: selected}


def patch_model_yaml(model_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Pick a single entry from base_model_config.yaml's `models`."""
    models = model_cfg.get("models", {}) or {}
    if model_name in models:
        model_cfg["models"] = {model_name: dict(models[model_name])}
        return model_cfg
    for k, v in models.items():
        if isinstance(v, Dict) and v.get("model_name") == model_name:
            model_cfg["models"] = {k: dict(v)}
            return model_cfg
    available = ", ".join(models.keys())
    print(f"[ERROR] Model '{model_name}' not found. Available model keys: {available}", file=sys.stderr)
    sys.exit(2)


def patch_infer_yaml(
    infer_cfg: Dict[str, Any], 
    output_dir: str, 
    server_url: str | None = None,
    all_override: bool = False,
    false_belief_override: bool = False
) -> Dict[str, Any]:
    """Patch inference yaml to set output directory, server URL, and override flags."""
    infer_cfg = dict(infer_cfg or {})
    infer_cfg["output_dir"] = output_dir
    if server_url:
        infer_cfg["server_url"] = server_url
    if all_override:
        infer_cfg["all_override"] = True
    if false_belief_override:
        infer_cfg["false_belief_override"] = True
    return infer_cfg


def run_cmd(cmd: List[str], cwd: Path | None = None) -> int:
    print("Running:", " ".join(shlex.quote(c) for c in cmd), f"(cwd={cwd or Path.cwd()})", flush=True)
    cp = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return cp.returncode


def _wait_for_http(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.getcode() == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _stream_process_output(proc: subprocess.Popen, prefix: str = "server") -> None:
    def _reader():
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                print(f"[{prefix}] {line}", end='')
        except Exception as e:
            print(f"[WARN] log stream error: {e}")
    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def start_env_server(host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u", "-m", "vagen.server.server",
        f"server.host={host}",
        f"server.port={port}",
        "use_state_reward=false",
    ]
    print("Starting env server:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    _stream_process_output(proc, prefix="server")
    health_url = f"http://{host}:{port}/health"
    if not _wait_for_http(health_url, timeout=40.0):
        raise RuntimeError(f"Env server failed to start at {health_url}")
    print(f"Env server is up at {health_url}")
    return proc


def stop_env_server(proc: subprocess.Popen) -> None:
    if not proc:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        print(f"[WARN] Failed to stop server: {e}")


def compute_combo_paths(
    output_root: str,
    model_name: str,
    exp_types: List[str],
    seed_range: tuple[int, int] | None,
    render_modes: List[str],
    enable_think: bool,
    data_dir: str,
) -> List[str]:
    """Find combo directory paths by constructing possible paths and filtering by seed.

    This function constructs possible directory paths based on the given parameters,
    then checks if config.json exists in those paths and filters by seed.

    Args:
        output_root: Base output directory
        model_name: Model name
        exp_types: List of experiment types ('active', 'passive')
        seed_range: Tuple of (start_seed, end_seed) or None
        render_modes: List of render modes ('vision', 'text')
        enable_think: Whether thinking is enabled
        data_dir: Data directory containing room data (kept for compatibility)
        proxy_agent: Proxy agent for passive mode

    Returns:
        List of combo directory paths that match the criteria
    """
    # data_dir is kept for backward compatibility but not used in new logic
    _ = data_dir

    # Determine seed range
    if seed_range:
        seed_start, seed_end = seed_range
    else:
        seed_start, seed_end = None, None

    combo_paths = []
    model_dir = os.path.join(output_root, get_model_name(model_name))

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Warning: Model directory does not exist: {model_dir}", file=sys.stderr)
        return combo_paths

    # Build think string based on enable_think
    think_str = "think" if enable_think else "nothink"

    # Get all room_hash directories (first level under model_dir)
    try:
        room_hash_dirs = [d for d in os.listdir(model_dir)
                         if os.path.isdir(os.path.join(model_dir, d))]
    except Exception as e:
        print(f"Warning: Failed to list directories in {model_dir}: {e}", file=sys.stderr)
        return combo_paths

    # Construct possible paths based on parameters
    for room_hash in room_hash_dirs:
        for render_mode in render_modes:
            for exp_type in exp_types:
                proxy_agent = set_proxy_agent(exp_type, render_mode)
                # Build path: model_dir/room_hash/render_mode/exp_type/think_str/[proxy_agent]
                if exp_type == "passive" and proxy_agent:
                    combo_path = os.path.join(model_dir, room_hash, render_mode,
                                             exp_type, think_str, proxy_agent)
                else:
                    combo_path = os.path.join(model_dir, room_hash, render_mode,
                                             exp_type, think_str)

                # Check if this path exists and has config.json
                config_path = os.path.join(combo_path, STATE_BASENAME)
                if not os.path.exists(config_path):
                    continue

                # Read config and check seed
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    # Check if seed exists in config
                    seed = config.get("seed")
                    if seed is None:
                        continue

                    # Filter by seed range if specified
                    if seed_range is not None:
                        if not (seed_start <= seed <= seed_end):
                            continue

                    # Add this path to results
                    combo_paths.append(combo_path)

                except Exception as e:
                    print(f"Warning: Failed to read config from {config_path}: {e}",
                          file=sys.stderr)
                    continue

    return combo_paths

def set_proxy_agent(exp_type: str, render_mode: str) -> str | None:
    """Determine proxy agent based on exp_type and render_mode."""
    if exp_type == 'passive':
        if render_mode == 'vision':
            return "scout"
        else:
            return "strategist"
    return None

def run_exploration_phase(args, seed_opts, server_url: str | None,
                         exp_types: List[str], render_modes: List[str]):
    """Run exploration phase: create dataset and run inference once.

    Note: All seeds are processed in a single run via seed_opts.
    Loops through all exp_type and render_mode combinations.
    """
    print("\n" + "="*60)
    print("PHASE: EXPLORATION")
    print("="*60 + "\n")

    base_env = Path(args.base_env)
    base_infer = Path(args.base_infer)
    base_model = Path(args.base_model)

    # Load room_config from base_env_config.yaml
    base_env_cfg = load_yaml(base_env)
    room_config = base_env_cfg.get("room_config")
    if room_config:
        print(f"Loaded room_config from {base_env}: {room_config}")

    # Generate unique run_id for this combination
    combo_run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    data_train = f"data/{combo_run_id}/train.parquet"
    data_test = f"data/{combo_run_id}/test.parquet"

    tmp_paths = build_tmp_paths(combo_run_id, "exploration")

    infer_cfg = load_yaml(base_infer)
    # Patch inference config with all_override and false_belief_override flags if specified
    patched_infer_cfg = patch_infer_yaml(
        infer_cfg,
        args.output_root,
        server_url=server_url,
        all_override=args.all_override,
        false_belief_override=args.false_belief_override,
    )
    model_cfg = load_yaml(base_model)
    model_cfg = patch_model_yaml(model_cfg, args.model_name)
    dump_yaml(model_cfg, tmp_paths["model"])
    dump_yaml(patched_infer_cfg, tmp_paths["infer"])

    # Loop through all combinations
    for exp_type in exp_types:
        for render_mode in render_modes:
            proxy_agent = set_proxy_agent(exp_type, render_mode)
            print(f"\n--- Running exploration: exp_type={exp_type}, render_mode={render_mode}, proxy_agent={proxy_agent} ---")
            # Create env config with current combination
            env_cfg = patch_env_yaml(exp_type, render_mode,
                                     seed_opts, args.enable_think, data_dir=args.data_dir,
                                     proxy_agent = proxy_agent, room_config=room_config,
                                     false_belief_exp=args.false_belief_exp, max_exp_steps=args.max_exp_steps,
                                     replay=args.replay)
            dump_yaml(env_cfg, tmp_paths["env"])
            
            # Create dataset
            print(f"Creating dataset for {exp_type} exploration...")
            rc = run_cmd([
                sys.executable, "-u", "-m", "vagen.env.create_dataset",
                "--yaml_path", str(tmp_paths["env"]),
                "--train_path", data_train,
                "--test_path", data_test,
                "--force_gen",
            ])
            if rc != 0:
                sys.exit(rc)
            
            # Run inference
            print(f"Running exploration inference...")
            val_path = data_test
            wandb_path_name = "spatial_gym"
            cmd = [
                sys.executable, "-u", "-m", "vagen.inference.run_inference",
                f"--inference_config_path={tmp_paths['infer']}",
                f"--model_config_path={tmp_paths['model']}",
                f"--val_files_path={val_path}",
                f"--wandb_path_name={wandb_path_name}",
            ]
            rc = run_cmd(cmd)
            if rc != 0:
                sys.exit(rc)
            
            print(f"Exploration completed for {exp_type} + {render_mode}")
    
    print(f"\nAll exploration combinations completed. Results in: {args.output_root}")


def run_phase(args, mode: str, seed_opts: tuple[int, int] | None = None,
                       exp_types: List[str] = None, render_modes: List[str] = None):
    """Run inference phase: build messages and run inference for evaluation or cogmap.
    
    Computes combo paths for all exp_type and render_mode combinations at once.
    
    Args:
        args: Command line arguments
        mode: 'eval', 'cogmap', or 'reeval'
        seed_opts: Seed range tuple
        exp_types: List of experiment types
        render_modes: List of render modes
    """
    print("\n" + "="*60)
    print(f"PHASE: {mode.upper()}")
    print("="*60 + "\n")
    
    # Get model name
    model_name = load_yaml(Path(args.base_model))['models'][args.model_name]['model_name']
    
    # Compute combo paths for all combinations at once
    print("Computing combo directory paths for all combinations...")
    all_combo_paths = compute_combo_paths(
        output_root=args.output_root,
        model_name=model_name,
        exp_types=exp_types,
        seed_range=seed_opts,
        render_modes=render_modes,
        enable_think=bool(args.enable_think),
        data_dir=args.data_dir,
    )
    print(all_combo_paths)
    
    if not all_combo_paths:
        print("[ERROR] No valid combo paths computed", file=sys.stderr)
        sys.exit(2)
    
    print(f"Found {len(all_combo_paths)} combo directories to process")
    
    # Special handling for reevaluate modes
    if mode == "reeval":
        reevaluate_combo_dirs(all_combo_paths)
        print(f"\nEvaluation re-evaluation completed.")
        return

    if mode == "cogmap_reeval":
        reevaluate_cogmaps_combo_dirs(all_combo_paths)
        print(f"\nCogmap re-evaluation completed.")
        return
    
    if mode == "cogmap_fb_reeval":
        reevaluate_cogmap_fb_combo_dirs(all_combo_paths)
        print(f"\nFalse belief cogmap re-evaluation completed.")
        return

    # Build kwargs for run_inference_for_combo_dirs based on mode
    inference_kwargs = {
        "combo_dirs": all_combo_paths,
        "model_config": load_yaml(Path(args.base_model))['models'][args.model_name],
        "mode": mode,
        "inference_mode": args.inference_mode,
        "image_dir": args.data_dir,
    }

    if mode == "eval":
        if args.eval_task_counts:
            eval_task_counts = json.loads(args.eval_task_counts)
        else:
            base_infer = Path(args.base_infer)
            infer_cfg = load_yaml(base_infer)
            eval_task_counts = infer_cfg.get("eval_task_counts")
            if eval_task_counts:
                print(f"Using eval_task_counts from inference_config.yaml: {eval_task_counts}")
            else:
                raise FileNotFoundError("eval_task_counts not found in inference_config.yaml")
        
        # Filter by --tasks if provided
        if args.tasks:
            eval_task_counts = {k: v for k, v in eval_task_counts.items() if k in args.tasks}
            print(f"Filtered tasks by --tasks: {eval_task_counts}")

        # Filter vision tasks if render_mode is text only
        # if render_modes == ["text"]:
        #     original_keys = list(eval_task_counts.keys())
        #     eval_task_counts = {k: v for k, v in eval_task_counts.items() if 'vision' not in k}
        #     if len(eval_task_counts) < len(original_keys):
        #         print(f"Filtered out vision tasks for text-only mode. Remaining: {eval_task_counts}")

        inference_kwargs.update({
            "eval_task_counts": eval_task_counts,
            "eval_override": args.eval_override,
            "eval_mode": args.eval_mode,
        })
    elif mode == "cogmap_fb":
        inference_kwargs.update({
            "cogmap_fb_override": args.cogmap_fb_override,
        })
    else:  # cogmap
        inference_kwargs.update({
            "cogmap_override": args.cogmap_override,
            "last_global_only": args.cogmap_last_global_only,
        })

    # Run inference
    run_inference_for_combo_dirs(**inference_kwargs)
    
    print(f"\n{mode.capitalize()} completed.")

def run_aggregation_phase(args):
    """Run aggregation phase: aggregate logs and images from previous runs."""
    print("\n" + "="*60)
    print("PHASE: AGGREGATION")
    print("="*60 + "\n")
    
    SpatialEnvLogger.log_each_env_info(
        output_dir= args.output_root,
        model_name = load_yaml(Path(args.base_model))['models'][args.model_name]['model_name'],
        save_images=True,
    )
    
    print("\nAggregation completed.")

def main():
    args = parse_args()

    # Determine false_belief_exp based on phase
    args.false_belief_exp = args.phase == 'explore_fb'

    # Stream prints when piping (e.g., "2>&1 | tee ...")
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    
    # Environment variables
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    
    # Check base config files exist
    base_infer = Path(args.base_infer)
    base_model = Path(args.base_model)
    if not base_infer.exists() or not base_model.exists():
        print(f"Base YAML missing: infer={base_infer.exists()} model={base_model.exists()}",
              file=sys.stderr)
        sys.exit(2)
    
    # Parse seed range
    seed_opts = None
    if args.seed_range:
        try:
            s, e = [int(x) for x in args.seed_range.split('-', 1)]
            seed_opts = (s, e)
        except Exception:
            print(f"[ERROR] Bad --seed_range '{args.seed_range}'. Use 'start-end'.", file=sys.stderr)
            sys.exit(2)
    else:
        seed_opts = (0, 0 + args.num - 1)
    
    exp_types = [x.strip() for x in args.exp_type.split(',')]
    render_modes = [x.strip() for x in args.render_mode.split(',')]
    
    try:
        server_url: str | None = None
        
        # Start server only for exploration phase
        if args.phase in ['explore', 'explore_fb', 'all'] and not args.no_server:
            actual_port = get_adaptive_port(args.server_port, 5000)
            if actual_port != args.server_port:
                print(f"Using port {actual_port} instead of requested {args.server_port}")
            server_proc = start_env_server(args.server_host, actual_port)
            server_url = f"http://{args.server_host}:{actual_port}"
        
        # Run requested phase(s)
        if args.phase in ['explore', 'explore_fb']:
            run_exploration_phase(args, seed_opts, server_url, exp_types, render_modes)
        elif args.phase == 'eval':
            run_phase(args, mode="eval", seed_opts=seed_opts,
                              exp_types=exp_types, render_modes=render_modes)
        elif args.phase == 'cogmap':
            run_phase(args, mode="cogmap", seed_opts=seed_opts,
                          exp_types=exp_types, render_modes=render_modes)
        elif args.phase == 'cogmap_fb':
            run_phase(args, mode="cogmap_fb", seed_opts=seed_opts,
                          exp_types=exp_types, render_modes=render_modes)
        elif args.phase == 'reeval':
            run_phase(args, mode="reeval", seed_opts=seed_opts,
                              exp_types=exp_types, render_modes=render_modes)
        elif args.phase == 'cogmap_reeval':
            run_phase(args, mode="cogmap_reeval", seed_opts=seed_opts,
                              exp_types=exp_types, render_modes=render_modes)
        elif args.phase == 'cogmap_fb_reeval':
            run_phase(args, mode="cogmap_fb_reeval", seed_opts=seed_opts,
                              exp_types=exp_types, render_modes=render_modes)
        elif args.phase == 'all':
            run_exploration_phase(args, seed_opts, server_url, exp_types, render_modes)
            run_phase(args, mode="eval", seed_opts=seed_opts,
                              exp_types=exp_types, render_modes=render_modes)
            # Run cogmap for active exp_types
            if 'active' in exp_types:
                active_exp_types = [e for e in exp_types if e == 'active']
                run_phase(args, mode="cogmap", seed_opts=seed_opts,
                                  exp_types=active_exp_types, render_modes=render_modes)
        run_aggregation_phase(args)
    
    except Exception as e:
        raise e
    
    finally:
        if args.phase in ['explore', 'explore_fb', 'all'] and not args.no_server:
            stop_env_server(server_proc)
    
    print("\n" + "="*60)
    print("ALL PHASES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
