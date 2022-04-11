import argparse
import numpy as np
np.random.seed(0)
from ruamel.yaml import YAML
import os
from models import SSMGRL

def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
    parser.add_argument('--sparse_adj', type=bool, default=False, help='sparse adjacency matrix')
    parser.add_argument('--iterater', type=int, default=10, help='iterater')
    parser.add_argument('--use_pretrain', type=bool, default=True, help='use_pretrain')
    parser.add_argument('--nb_epochs', type=int, default=1500, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=5, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--dropout', type=int, default=0.2, help='dropout')
    parser.add_argument('--lambd0', default=0.0001, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd1', default=0.009, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd2', default= 0.008, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd3', default=0.0001, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd4', default=0.001, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd5', default=0.001, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=10, help='')
    parser.add_argument('--w_loss3', type=float, default=0.01, help='')
    parser.add_argument('--w_loss4', type=float, default=0.1, help='')
    parser.add_argument('--w_loss5', type=float, default=0.1, help='')

    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

def main():
    args = get_args(
        model_name="SSMGRL",
        dataset="acm",
        custom_key="Node",  # Node: node classification  Clu: clustering   Sim: similarity
    )
    if args.dataset == "imdb" or args.dataset == "acm" :
        args.length = 2
    else:
        args.length = 3
    printConfig(args)
    embedder = SSMGRL(args)
    macro_f1s, micro_f1s, k1, st = embedder.training()
    return macro_f1s, micro_f1s, k1, st


if __name__ == '__main__':
    macro_f1s, micro_f1s, k1, st = main()
