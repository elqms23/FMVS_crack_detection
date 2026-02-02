from config import build_argparser, make_config
from pipeline import run
2

def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = make_config(args)
    run(cfg)


if __name__ == "__main__":
    main()

# python main.py --video ./videos/... --show --ranges ""