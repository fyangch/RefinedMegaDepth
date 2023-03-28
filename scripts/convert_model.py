"""Script to convert a model from .txt files to .bin files and vice versa."""

import argparse

import megadepth.utils.read_write_model as model_io


def main(args: argparse.Namespace) -> None:
    """Convert a model from .txt files to .bin files and vice versa.

    Args:
        args: The parsed command line arguments.
    """
    cameras, images, points3D = model_io.read_model(args.model_path)
    print(f"Writing model as .{args.save_as}")
    model_io.write_model(
        cameras=cameras,
        images=images,
        points3D=points3D,
        path=args.model_path,
        ext=f".{args.save_as}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the input model.",
    )
    parser.add_argument(
        "--save-as",
        type=str,
        choices=["txt", "bin"],
        default="bin",
        help="Format of the output model.",
    )
    args = parser.parse_args()

    main(args)
