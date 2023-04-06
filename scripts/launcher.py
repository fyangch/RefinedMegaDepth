"""Launcher script to submit jobs to the ethz euler cluster."""

import argparse
import re
import subprocess
import sys
from math import ceil

models = {
    "3090": "rtx_3090",
    "1080": "rtx_1080",
    "2080Ti": "rtx_2080_ti",
    "TITAN": "titan_rtx",
    "V100": "v100",
}


def main(args):
    """Submit a job to the cluster."""
    job_id = None

    for i in range(args.chain):
        cmd = f"{'srun --pty' if args.interactive else 'sbatch'} --time={args.time}:00 "
        cmd += f"--nodes=1 --ntasks={max(1, args.gpus)} "
        cmd += f"--cpus-per-task={int(args.cpus/max(1, args.gpus))} "
        jobname = args.name + str(i) if (args.chain > 1) else args.name
        cmd += f"-J {jobname} "
        if i > 0:
            cmd += f" -d afterany:{job_id} "
        cmd += f"--mem-per-cpu={ceil(args.mem/args.cpus)} "
        cmd += f"--tmp={args.scratch} "

        if args.gmod is not None:
            cmd += f"--gpus={models[args.gmod]}:{args.gpus} "
        elif args.gmem is not None:
            cmd += f"--gres=gpumem:{args.gmem}m --gpus={args.gpus} "

        if args.mail:
            if not args.mail_user:
                raise ValueError("Please provide an email address with --mail-user")
            cmd += f"--mail-type={args.mail_type} "
            cmd += f"--mail-user={args.mail_user} "

        wrap_cmd = " ".join(args.command)
        cmd += f'--wrap="{wrap_cmd}"'
        print(cmd)
        exit()  # remove this line to actually submit the job
        OUT = sys.stdout if args.interactive else subprocess.PIPE
        ERR = sys.stderr if args.interactive else subprocess.STDOUT
        try:
            ret = subprocess.run(cmd, shell=True, check=True, stdout=OUT, stderr=ERR, text=True)
            out_s = ret.stdout
        except subprocess.CalledProcessError as e:
            out_s = e.output
            print(e)
        except Exception as e:
            print(e)
            raise e
        if not args.interactive:
            print(out_s)
            (job_id,) = re.findall(r"Submitted batch job (\d+)", out_s)
            print(job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", "-c", type=int, default=20)
    parser.add_argument("--mem", "-m", type=int, default=60_000, help="Total memory (NOT per core)")
    parser.add_argument("--name", "-n", type=str, default="job")
    parser.add_argument("--scratch", type=int, default=0_000)
    parser.add_argument("--time", type=str, default="24:00")
    parser.add_argument("--gpus", "-g", type=int, default=1)
    parser.add_argument("--gmod", type=str, choices=list(models), help="GPU model", default=None)
    parser.add_argument("--gmem", type=int, default=10_240, help="GPU memory")
    parser.add_argument("--chain", "-ch", type=int, default=1)
    parser.add_argument("--interactive", "-I", action="store_true")
    parser.add_argument("--mail", action="store_true", help="Send email on job status")
    parser.add_argument(
        "--mail-type",
        type=str,
        default="ALL",
        help="Email type (ALL, BEGIN, END, FAIL, REQUEUE, NONE)",
    )
    parser.add_argument("--mail-user", type=str, help="Email address")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    main(parser.parse_args())
