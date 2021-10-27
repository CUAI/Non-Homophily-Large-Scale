import argparse
import os
import glob
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

import numpy as np

from logger import SimpleLogger
from dataset import load_nc_dataset
from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import normalize, gen_normalized_adjs
from data_utils import evaluate, eval_acc, eval_rocauc, load_fixed_splits

import optuna


def main():
    parser = argparse.ArgumentParser(description='C&S Hyperparameters Tuning')
    parser.add_argument('--dataset', type=str, default='fb100')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--directed', action='store_true',
                    help='set to not symmetrize adjacency')
    parser.add_argument('--hops', type=int, default=1,
                    help='power of adjacency matrix for certain methods')
    parser.add_argument('--rocauc', action='store_true', 
                    help='set the eval function to rocauc')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cs_fixed', action='store_true', help='use FDiff-scale')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--evaluate', action='store_true', 
                    help='evaluate base mlp and c&s for certain hyperparameters')
    parser.add_argument('--A1', type=str, default='DAD', choices=['DAD', 'DA', 'AD'])
    parser.add_argument('--A2', type=str, default='DAD', choices=['DAD', 'DA', 'AD'])
    parser.add_argument('--alpha1', type=float, default=0.5)
    parser.add_argument('--alpha2', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--trials', type=int, default=100)
    args = parser.parse_args()

    # consistent data splits, see data_utils.rand_train_test_idx
    np.random.seed(0)

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.cpu:
        device = torch.device('cpu')

    dataset = load_nc_dataset(args.dataset, args.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
        eval_func = eval_rocauc
    else:
        eval_func = eval_acc

    model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}-None'
    model_dir = f'models/{model_path}'
    print(model_dir)
    model_outs = glob.glob(f'{model_dir}/*.pt')
    
    name = f'{args.dataset}-{args.sub_dataset}-{args.hops}' if args.sub_dataset \
        else f'{args.dataset}-{args.hops}'
    if args.cs_fixed: name += '-f'

    if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
        split_idx_lst = [dataset.get_idx_split() for _ in model_outs]
    else:
        split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)
    
    norm_adjs = {}
    norm_adjs['DAD'], norm_adjs['DA'], norm_adjs['AD'] = gen_normalized_adjs(dataset)
    
    # get base mlp and c&s results for certain hyperparameters
    if args.evaluate:
        alpha1 = args.alpha1
        alpha2 = args.alpha2
        A1 = args.A1
        A2 = args.A2
        
        base_logger = SimpleLogger('evaluate params', [], 2)
        logger = SimpleLogger('evaluate params', [], 2)
        for run, model_out in enumerate(model_outs):
            split_idx = split_idx_lst[run]
            out = torch.load(model_out, map_location='cpu')
            
            base_result = evaluate(None, dataset, split_idx, eval_func, out)
            base_logger.add_result(run, (), (base_result[1], base_result[2]))
        
            if args.cs_fixed:
                _, out_cs = double_correlation_fixed(dataset.label, out, split_idx, 
                    norm_adjs[A1], alpha1, 50, norm_adjs[A2], alpha2, 50, args.scale, args.hops)  
            else:
                _, out_cs = double_correlation_autoscale(dataset.label, out, split_idx, 
                    norm_adjs[A1], alpha1, 50, norm_adjs[A2], alpha2, 50, args.hops)
            result = evaluate(None, dataset, split_idx, eval_func, out_cs)
            logger.add_result(run, (), (result[1], result[2]))
        print('Base valid -> Base test')
        base_res = base_logger.display()
        print('Final valid -> Final test')
        res = logger.display()
        
        base_valid = f'{base_res[:, 0].mean():.3f} ± {base_res[:, 0].std():.3f}'
        base_test = f'{base_res[:, 1].mean():.3f} ± {base_res[:, 1].std():.3f}'
        valid = f'{res[:, 0].mean():.3f} ± {res[:, 0].std():.3f}'
        test = f'{res[:, 1].mean():.3f} ± {res[:, 1].std():.3f}'
        with open("cs_eval.txt", "a+") as write_obj:
            write_obj.write(f"{name}," +
                            f"# base {base_valid} -> {base_test}\n" +
                            f"# post {valid} -> {test}\n")
    else:
        def objective(trial):
            alpha1 = trial.suggest_uniform("alpha1", 0.0, 1.0)
            alpha2 = trial.suggest_uniform("alpha2", 0.0, 1.0)
            A1 = trial.suggest_categorical('A1', ['DAD', 'DA', 'AD'])
            A2 = trial.suggest_categorical('A2', ['DAD', 'DA', 'AD'])

            if args.cs_fixed:
                scale = trial.suggest_loguniform("scale", 0.1, 10.0)

            logger = SimpleLogger('evaluate params', [], 2)
            for run, model_out in enumerate(model_outs):
                split_idx = split_idx_lst[run]
                out = torch.load(model_out, map_location='cpu')
                if args.cs_fixed:
                    _, out_cs = double_correlation_fixed(dataset.label, out, split_idx, 
                        norm_adjs[A1], alpha1, 50, norm_adjs[A2], alpha2, 50, scale, args.hops)  
                else:
                    _, out_cs = double_correlation_autoscale(dataset.label, out, split_idx, 
                        norm_adjs[A1], alpha1, 50, norm_adjs[A2], alpha2, 50, args.hops)
                result = evaluate(None, dataset, split_idx, eval_func, out_cs)
                logger.add_result(run, (), (result[1], result[2]))
            res = logger.display()

            trial.set_user_attr('valid', f'{res[:, 0].mean():.3f} ± {res[:, 0].std():.3f}')
            trial.set_user_attr('test', f'{res[:, 1].mean():.3f} ± {res[:, 1].std():.3f}')

            return res[:, 0].mean()


        study = optuna.create_study(study_name=f'{name}', 
            storage=f'sqlite:///{name}.db', direction="maximize", load_if_exists=True)
        study.optimize(objective, n_trials=args.trials)
        best_attr = study.best_trial.user_attrs
        print('Final valid -> Final test')
        print('{valid} -> {test}'.format(**best_attr))
        print(f'Best params: {study.best_params}')

        with open("cs_hparams.txt", "a+") as write_obj:
            write_obj.write(f"{name}," +
                            f"{study.best_params} \n" + 
                            "# {valid} -> {test}\n".format(**best_attr))


if __name__ == "__main__":
    main()