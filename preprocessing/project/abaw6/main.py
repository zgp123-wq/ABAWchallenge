import sys

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hello, world.')

    parser.add_argument('-python_package_path', default='/home/data/lrd/zgp/abaw/abaw5_preprocessing', type=str,
                    help='The path to the entire repository.')
    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    from project.abaw5.preprocessing import PreprocessingABAW5
    from project.abaw5.configs import config

    pre = PreprocessingABAW5(config)
    pre.generate_per_trial_info_dict()
    pre.prepare_data()


