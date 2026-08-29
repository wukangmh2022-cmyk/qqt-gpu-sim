#!/usr/bin/env python3
"""Create one loader-valid warm-start checkpoint from a parameters-only snapshot.

This is intentionally a warm start at iteration zero. It preserves model weights but
initializes the optimizer, environments, and random streams for the target rank.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax


def tree_stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def fail(message):
    raise SystemExit(f"WARMSTART_ERROR: {message}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True,
                        help="parameters-only snapshot to warm start from")
    parser.add_argument("--ckpt-dir", required=True,
                        help="new checkpoint directory for this rank")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--expected-local-devices", type=int, default=2)
    parser.add_argument("--arch", default="transformer")
    parser.add_argument("--embed", type=int, default=392)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-factor", type=float, default=4.0)
    parser.add_argument("--num-envs", type=int, default=32768)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--minibatch", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", required=True)
    parser.add_argument(
        "--level-weights",
        default="empty=0.05,\u529f\u592b=0.1,\u6bd4\u6b66=0.15",
    )
    parser.add_argument("--crate-reward-coef", type=float, default=0.5)
    parser.add_argument("--crate-reward-anneal-steps", type=int,
                        default=30000000000)
    parser.add_argument("--explore-reward-coef", type=float, default=0.01)
    parser.add_argument("--explore-reward-anneal-steps", type=int,
                        default=30000000000)
    parser.add_argument("--brick-reward-coef", type=float, default=0.05)
    parser.add_argument("--reward-anneal-k", type=float, default=1.2)
    parser.add_argument("--reward-anneal-step-offset", type=int, default=0,
                        help="completed global environment steps from the parameter source; "
                             "preserves fixed reward annealing without faking optimizer resume")
    parser.add_argument("--lsgd-k", type=int, default=256)
    parser.add_argument("--lsgd-mode", choices=("param", "grad"),
                        default="param")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_project_root():
    """Make this standalone helper work from both the repo and a staged run root."""
    candidates = (Path.cwd(), Path(__file__).resolve().parent)
    for candidate in candidates:
        if (candidate / "jax_bomb").is_dir():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    fail("cannot find a sibling jax_bomb package; run from the staged run directory")


def load_params(path):
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        fail(f"cannot load parameters snapshot {path}: {exc}")
    if isinstance(payload, dict) and "params" in payload:
        payload = payload["params"]
    return payload


def check_param_shapes(params, expected):
    leaves, tree = jax.tree.flatten(params)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    if tree != expected_tree:
        fail("parameters snapshot tree does not match the target Transformer")
    if len(leaves) != len(expected_leaves):
        fail("parameters snapshot leaf count does not match the target Transformer")
    for index, (leaf, reference) in enumerate(zip(leaves, expected_leaves)):
        if np.shape(leaf) != tuple(reference.shape):
            fail(
                "parameters snapshot shape mismatch at leaf "
                f"{index}: got {np.shape(leaf)}, expected {tuple(reference.shape)}"
            )


def param_digest(params):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(params):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.world_size < 1:
        fail("world size must be positive")
    if not 0 <= args.rank < args.world_size:
        fail(f"rank {args.rank} is outside world size {args.world_size}")
    if args.reward_anneal_step_offset < 0:
        fail("reward anneal step offset must be non-negative")
    if not os.path.isfile(args.params):
        fail(f"parameters snapshot is missing: {args.params}")
    if not os.path.isfile(args.levels):
        fail(f"levels file is missing: {args.levels}")

    find_project_root()
    n_local = jax.local_device_count()
    if n_local != args.expected_local_devices:
        fail(
            f"rank {args.rank} sees {n_local} local JAX devices; expected "
            f"{args.expected_local_devices}"
        )
    n_total = n_local * args.world_size
    envs_per = args.num_envs // n_total
    mb_local = args.minibatch // n_total
    if envs_per < 1 or mb_local < 1:
        fail("num-envs or minibatch is too small for the requested replica count")
    effective_num_envs = envs_per * n_total
    effective_minibatch = mb_local * n_total

    # Keep environment initialization identical to multicard_train.main().
    from jax_bomb import levels
    levels.set_active(args.levels, weights=args.level_weights)
    from jax_bomb.jax_env import H, N_OBS_CH, RADIUS, W, init_batch
    from jax_bomb.jax_net import init_net

    source = load_params(args.params)
    expected = init_net(
        jrandom.PRNGKey(args.seed + 9999),
        args.arch,
        N_OBS_CH,
        H,
        W,
        embed=args.embed,
        depth=args.depth,
        patch=args.patch,
        heads=args.heads,
        ff_factor=args.ff_factor,
    )
    check_param_shapes(source, expected)
    params = jax.tree.map(jnp.asarray, source)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    states = tree_stack([
        init_batch(
            jrandom.PRNGKey(args.seed * 1000 + args.rank * n_local + local_device),
            envs_per,
        )
        for local_device in range(n_local)
    ])
    keys = jrandom.split(jrandom.PRNGKey(args.seed * 7919 + args.rank), n_local)

    cfg = {
        "arch": args.arch,
        "embed": args.embed,
        "depth": args.depth,
        "patch": args.patch,
        "heads": args.heads,
        "ff_factor": args.ff_factor,
        "num_envs": effective_num_envs,
        "num_steps": args.num_steps,
        "minibatch": effective_minibatch,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "radius": RADIUS,
        "n_total": n_total,
        "map": f"{H}x{W}",
        "levels": os.path.basename(args.levels),
        "crate_coef": args.crate_reward_coef,
        "crate_anneal": args.crate_reward_anneal_steps,
        "explore_coef": args.explore_reward_coef,
        "explore_anneal": args.explore_reward_anneal_steps,
        "brick_coef": args.brick_reward_coef,
        "anneal_k": args.reward_anneal_k,
        "reward_anneal_step_offset": args.reward_anneal_step_offset,
        "envs_per": envs_per,
        "mb_local": mb_local,
        "lsgd_k": args.lsgd_k,
        "lsgd_mode": args.lsgd_mode,
        "lsgd_bf16": False,
        "lsgd_sync_state": False,
    }
    payload = {
        "it": 0,
        "cfg": cfg,
        "params": jax.tree.map(np.asarray, params),
        "opt_state": jax.tree.map(np.asarray, opt_state),
        "states": jax.tree.map(np.asarray, states),
        "keys": np.asarray(keys),
    }
    os.makedirs(args.ckpt_dir, exist_ok=True)
    target = os.path.join(args.ckpt_dir, f"ckpt_00000000_r{args.rank}.pkl")
    if os.path.exists(target) and not args.overwrite:
        fail(f"checkpoint already exists: {target}")
    descriptor = {
        "kind": "parameter_warm_start",
        "source": os.path.basename(args.params),
        "source_iteration": 889,
        "source_global_steps": args.reward_anneal_step_offset,
        "checkpoint_iteration": 0,
        "optimizer_environment_rng": "fresh",
        "dynamic_reward_annealing": "restarts from first rollout",
        "rank": args.rank,
        "world_size": args.world_size,
        "local_devices": n_local,
        "global_replicas": n_total,
        "envs_per_replica": envs_per,
        "minibatch_per_replica": mb_local,
        "param_digest": param_digest(params),
    }
    fd, temporary = tempfile.mkstemp(prefix=".warmstart_", suffix=".pkl", dir=args.ckpt_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    info_path = os.path.join(args.ckpt_dir, f"warmstart_r{args.rank}.json")
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(descriptor, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "WARMSTART_OK "
        f"rank={args.rank} checkpoint={target} replicas={n_total} "
        f"envs_per={envs_per} mb_local={mb_local} "
        f"digest={descriptor['param_digest']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
