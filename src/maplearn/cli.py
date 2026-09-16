import argparse


def main():
    parser = argparse.ArgumentParser(description="How a Transformer Learns a Map")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--topology", choices=["wrap", "bounded", "portal"], default="wrap")
    train.add_argument("--out", required=True)
    train.add_argument("--size", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--routes", type=int, default=250000)
    train.add_argument("--eval-routes", type=int, default=5000)
    train.add_argument("--max-moves", type=int, default=12)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--checkpoints", type=int, default=40)
    train.add_argument("--lr", type=float, default=0.001)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--routes", type=int, default=2000)
    analyze.add_argument("--layer", type=int, choices=[0, 1], default=1)
    analyze.add_argument("--ridge", type=float, default=1.0)
    analyze.add_argument("--strength", type=float, default=1.0)
    for command in (train, analyze):
        command.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "train":
        from .train import train

        train(args)
    else:
        from .analyze import analyze

        analyze(args)


if __name__ == "__main__":
    main()
