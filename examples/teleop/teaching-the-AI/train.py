
import torch
from torch.utils.data import DataLoader

from robot_ai import RobotData
from robot_ai import Policy
from robot_ai import split_episodes


#validation pass (no backward/step)

def evaluate(policy, val_loader, device, use_state):
    policy.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            state = batch["state"].to(device) if use_state else None
            actions = batch["actions"].to(device)

            guess = policy(images, state)
            truth = policy.normalize_actions(actions)
            loss = torch.nn.functional.l1_loss(guess, truth)

            total_loss += loss.item()
            n_batches += 1

    policy.train()
    return total_loss / n_batches


def main():
    #make it repeatable
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"using device: {device}")

    data_folder = "data/pick-cube"  # <-- point this at your prepared dataset
    use_state = True      # flip to False to train image-only (state_size=0)

    #splits takes
    train_eps, val_eps = split_episodes(data_folder, val_fraction=0.15, seed=0)
    print(f"train: {len(train_eps)} takes, val: {len(val_eps)} takes")

    #builds datasets
    train_data = RobotData(data_folder, chunk=15, episodes=train_eps)
    val_data = RobotData(data_folder, chunk=15, episodes=val_eps)
    print(f"train examples: {len(train_data)}, val examples: {len(val_data)}")

    sample = train_data[0]
    print("image shape:  ", sample["image"].shape)
    print("state shape:  ", sample["state"].shape)
    print("actions shape:", sample["actions"].shape)

    #DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_data, batch_size=64, shuffle=False, num_workers=4
    )

    #build policy (stats from TRAIN data only)
    mean, std = train_data.action_stats()
    state_mean, state_std = train_data.state_stats()

    policy = Policy(
        action_size=train_data.action_size,
        chunk=15,
        action_mean=mean,
        action_std=std,
        state_size=train_data.state_size if use_state else 0,
        state_mean=state_mean,
        state_std=state_std,
    )
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy has {n_params:,} parameters")

    #optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    # loop
    n_epochs = 30      # ~490 steps/epoch in the tutorial's reference run
    log_every = 50     # steps
    val_every = 400    # steps

    best_val_loss = float("inf")
    step = 0

    for epoch in range(n_epochs):
        for batch in train_loader:
            policy.train()

            images = batch["image"].to(device)
            state = batch["state"].to(device) if use_state else None
            actions = batch["actions"].to(device)

            guess = policy(images, state)
            # normalize the truth before comparingm, the policy works in
            # normalized units. Skipping this is the #1 silent bug: loss
            # looks great, robot behavior is nonsense.
            truth = policy.normalize_actions(actions)
            loss = torch.nn.functional.l1_loss(guess, truth)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_every == 0:
                print(f"epoch {epoch:3d} step {step:6d}  train_loss {loss.item():.4f}")

            if step % val_every == 0:
                val_loss = evaluate(policy, val_loader, device, use_state)
                print(f"epoch {epoch:3d} step {step:6d}  val_loss   {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    policy.save("my_policy.pt")
                    print("  -> new best val_loss, checkpoint saved")

            step += 1
    policy.save("my_policy.pt")
    print("done. saved final policy to my_policy.pt")


if __name__ == "__main__":
    main()