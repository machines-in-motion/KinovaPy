#!/usr/bin/env python
"""Mark your own homework: is the policy you trained any good?

    python tools/check_policy.py --policy my_policy.pt

It replays takes the policy has never seen, compares what it predicts against
what the human actually did, and compares BOTH against a lazy baseline that
always guesses the average action. A policy that cannot beat the lazy baseline
has learned nothing at all, however pretty its training curve looked.

Writes `prediction.png` next to the policy: your prediction (line) on top of the
human's (dots), for one whole take.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_ai import Policy, RobotData, pick_device, split_episodes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, required=True, help="the .pt file your training script saved")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "pick-cube")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="must match what you used while training")
    parser.add_argument("--seed", type=int, default=0, help="must match what you used while training")
    parser.add_argument("--episode", type=int, default=None, help="which held-out take to draw")
    args = parser.parse_args()

    device = pick_device()
    policy = Policy.load(args.policy, device=device)
    print(f"\npolicy      : {args.policy}")
    print(f"predicts    : {policy.chunk} moments x {policy.action_size} numbers")
    print(f"feels       : {policy.state_size} numbers about the robot"
          + ("" if policy.state_size else "  (pictures only)"))
    print(f"running on  : {device}")

    _, val_episodes = split_episodes(args.data, val_fraction=args.val_fraction, seed=args.seed)
    data = RobotData(args.data, chunk=policy.chunk, episodes=val_episodes)
    if policy.state_size and policy.state_size != data.state_size:
        raise SystemExit(
            f"the policy feels {policy.state_size} numbers but this data folder has {data.state_size}"
        )
    if policy.action_size != data.action_size:
        raise SystemExit(
            f"policy predicts {policy.action_size} numbers but the data has {data.action_size}"
        )
    print(f"testing on  : {len(val_episodes)} takes it has never seen ({len(data)} moments)\n")

    # --- how wrong is it, compared to just guessing the average? -------------
    loader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=2)
    mean, _ = data.action_stats()
    lazy = torch.as_tensor(mean, device=device)

    total_error = torch.zeros(data.action_size, device=device)
    lazy_error = torch.zeros(data.action_size, device=device)
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            truth = batch["actions"].to(device)
            state = batch["state"].to(device) if policy.state_size else None
            guess = policy.unnormalize_actions(policy(images, state))
            total_error += (guess - truth).abs().mean(dim=1).sum(dim=0)
            lazy_error += (lazy - truth).abs().mean(dim=1).sum(dim=0)
            count += images.shape[0]
    total_error = (total_error / count).cpu().numpy()
    lazy_error = (lazy_error / count).cpu().numpy()

    print(f"{'action number':<20s} {'your error':>11s} {'lazy guess':>11s} {'verdict':>12s}")
    verdicts = []
    for k, name in enumerate(data.action_names):
        better = lazy_error[k] / max(total_error[k], 1e-9)
        verdict = "learned" if better > 1.15 else ("no better" if better > 0.95 else "WORSE")
        verdicts.append(better)
        print(f"{name:<20s} {total_error[k]:11.4f} {lazy_error[k]:11.4f} {verdict:>12s}")

    overall = float(np.mean(verdicts))
    print(f"\noverall, your policy is {overall:.2f}x better than guessing the average.")
    if overall < 1.1:
        print("  -> that is basically nothing. Train longer, or check that you normalised")
        print("     the actions, or that you did not accidentally shuffle the pictures.")
    elif overall < 1.5:
        print("  -> it has learned something real, but not much. Worth more training.")
    else:
        print("  -> it has genuinely learned the task. Try it on the robot (tutorial 04).")

    # --- draw one take -------------------------------------------------------
    episode = args.episode if args.episode is not None else val_episodes[0]
    single = RobotData(args.data, chunk=policy.chunk, episodes=[episode])
    truths, guesses = [], []
    for i in range(len(single)):
        sample = single[i]
        state = sample["state"][None].to(device) if policy.state_size else None
        with torch.no_grad():
            out = policy.unnormalize_actions(policy(sample["image"][None].to(device), state))[0]
        guesses.append(out[0].cpu().numpy())  # the very next moment only
        truths.append(sample["actions"][0].numpy())
    truths, guesses = np.stack(truths), np.stack(guesses)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - no picture)")
        return
    names = single.action_names
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.3 * len(names)), sharex=True)
    for k, name in enumerate(names):
        axes[k].plot(truths[:, k], ".", markersize=2, label="human")
        axes[k].plot(guesses[:, k], linewidth=1, label="policy")
        axes[k].set_ylabel(name, rotation=0, ha="right", fontsize=8)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel(f"moment in take {episode} (never seen during training)")
    fig.tight_layout()
    out = args.policy.parent / "prediction.png"
    fig.savefig(out, dpi=110)
    print(f"\npicture: {out}\n")


if __name__ == "__main__":
    main()
