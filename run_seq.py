import argparse

from recbole.quick_start import run_recbole
from recbole.utils import init_seed

# datasets options: Amazon_All_Beauty   Amazon_Clothing_Shoes_and_Jewelry   Amazon_Sports_and_Outdoors
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CFIT4SRec', help='name of models')
    # parser.add_argument('--model', '-m', type=str, default='WaveRec', help='name of models')
    # parser.add_argument('--model', '-m', type=str, default='SASRec', help='name of models')
    # parser.add_argument('--dataset', '-d', type=str, default='ml-1m', help='name of datasets')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='seq.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
