import argparse
import glob
import os
import struct
from pathlib import Path

from tensorboard.compat.proto import event_pb2


def iter_event_records(event_path):
    with open(event_path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f"Truncated event header in {event_path}")

            (length,) = struct.unpack("<Q", header)
            handle.read(4)  # masked CRC for length
            data = handle.read(length)
            handle.read(4)  # masked CRC for payload
            if len(data) != length:
                raise ValueError(f"Truncated event payload in {event_path}")

            event = event_pb2.Event()
            event.ParseFromString(data)
            yield event


def main():
    parser = argparse.ArgumentParser(description="Print scalar values from TensorBoard event files.")
    parser.add_argument("logdir", help="A TensorBoard run directory, for example ckpt/Jun10_02-38-09.")
    parser.add_argument("--plot", help="Optional output PNG path for loss curves.")
    args = parser.parse_args()

    event_files = sorted(glob.glob(os.path.join(args.logdir, "events.out.tfevents.*")))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files found in {args.logdir}")

    scalars = {}
    for event_path in event_files:
        for event in iter_event_records(event_path):
            if not event.summary:
                continue
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    scalars.setdefault(value.tag, []).append((event.step, value.simple_value))

    for tag in sorted(scalars):
        print(f"[{tag}]")
        for step, value in scalars[tag]:
            print(f"step={step}\tvalue={value:.6f}")

    if args.plot:
        import matplotlib.pyplot as plt

        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        if "train_loss" in scalars:
            steps, values = zip(*scalars["train_loss"])
            ax.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label="train_loss")
        if "validation_loss" in scalars:
            steps, values = zip(*scalars["validation_loss"])
            ax.plot(steps, values, marker="s", linewidth=1.8, markersize=5, label="validation_loss")
        ax.set_title(os.path.basename(os.path.normpath(args.logdir)))
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()
