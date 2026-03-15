import os
from argparse import ArgumentParser

dark_scenes = ['bikes','gardenlights','nightstreet','notchbush']
# dark_scenes = []

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="output/raw")
parser.add_argument("--n_views", type=int, default=6, help="number of training views")
args, _ = parser.parse_known_args()

all_scenes = dark_scenes

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--raw','-raw', required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = f" --quiet --eval --test_iterations -1 -r 4 --n_views {args.n_views} --lambda_dist 20"
    for scene in dark_scenes:
        source = args.raw + "/" + scene
        print(
            "python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in dark_scenes:
        all_sources.append(args.raw + "/" + scene)

    common_args = f" --quiet --eval --skip_train --n_views {args.n_views} --num_cluster 50 --depth_trunc 30 "

    for scene, source in zip(all_scenes, all_sources):
        os.system(
            "python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
