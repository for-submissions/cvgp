
import argparse
import random
from collections import defaultdict
import os

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from datasets import load_dataset
from utils import (
    train_one_epoch,
    evaluate_model,
    cache_epoch_results,
    get_best_model,
    check_early_stop,
    save_epoch_stats,
    prepare_fold,
    cache_fold_results,
    save_fold_stats,
    get_best_model_back
    )

# GP and loss models
from models.model_ExactGP import ExactGP
from models.model_CVTGP import CVTGP
from models.model_RandomCVTGP import RandomCVTGP
# from models.model_EigenChooseCVTGP import EigenChooseCVTGP
# from models.model_EigenSparseCVTGP import EigenSparseCVTGP
from models.model_SVGP import SVGP #Hensman et al.
from models.model_SparseGP import SparseGP #Titsias
from models.model_SparseGP2 import SparseGP2 #Titsias
from models.model_PPGPR import PPGPR #Jankowiak et al.

# Environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument(
        '--device', default='cuda', type=str,
        help='device to train the model.'
        )
    #optimization args
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate.'
                        )
    parser.add_argument('--wd', default=0, type=float,
                        help='weight decay.'
                        )
    parser.add_argument('--epochs', default=int(1e5), type=int,
                        help='number of training epochs.'
                        )
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size.'
                        )
    parser.add_argument('--ip_size', default=100, type=int,
                        help='inducing point / coreset size. we always use \
                        100 throughout our experiments.'
                        )
    #model args
    parser.add_argument('--dtype', default='float32', type=str,
                        help='dtype to run experiments on. we use float32 \
                        for all models during our experiments. sparsegp can\
                        be numerically instable. in such cases, we switch to\
                        float64 for sparsegp.'
                        )
    parser.add_argument('--kernel', default='rbf', type=str,
                        choices=['rbf'],
                        help='GP kernel to use. \
                        we use rbf in the paper'
                        )
    parser.add_argument('--loss', default='SVGP', type=str,
                        choices=[
                            'ExactGP', 'CVTGP', 'RandomCVTGP',
                            'SVGP', 'PPGPR',
                            'SparseGP', 'SparseGP2',
                            #below not introduced
                            'EigenSparseCVTGP','FullSparseCVTGP',
                            'EigenChooseCVTGP', 
                            ],
                        help='the learning algorithm.'
                        )
    #data, fold, tune, metric args
    parser.add_argument('--cv_folds', default=5, type=int,
                        help='if you want to plot shapes use cv_folds=1'
                        )
    parser.add_argument('--dataset', default='bike', type=str,
                        choices=[
                            #synthetic datasets
                            'synthetic1','synthetic2','synthetic3',
                            'synthetic4', 'synthetic5', 
                            #real-world datasets
                            'wine', 'bike',
                            'parkinsons', 'protein', 'skillcraft',
                            'slice', '3droad', 'buzz', 'houseelectric',
                            'song', 'song',
                            #bernoulli
                            'moons'
                            ]
                        )
    parser.add_argument('--preprocess', default=True,
                        help='convert to action="store_false" if \
                        using a terminal and not running on an IDE.'
                        )
    parser.add_argument('--check_stop', default=150, type=int,
                        help='check early stop every 150 epochs.'
                        )
    parser.add_argument('--early_stop', default=20, type=int,
                        help='stop if no improvement for 20 checks.'
                        )
    parser.add_argument('--save_metric', default='rmse', type=str,
                        help='save with respect to rmse.'
                        )
    parser.add_argument('--fh', action='store_true',
                        help='fix hyperparameters during optimization.'
                        )
    parser.add_argument('--fi', action='store_true',
                        help='fix inducing points during optimization.'
                        )
    args = parser.parse_args()

    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    fold_results = defaultdict(lambda: defaultdict(list))

    dtype = {
        'float64': torch.double,
        'float32': torch.float,
    }[args.dtype]

    data, dtypes = load_dataset(
        dataset=args.dataset
        )
    x, y = data

    if 'rmse' in args.save_metric.lower():
        METRIC = 'RMSE'
    else:
        METRIC = 'Loss'

    d_in = x.shape[1]
    n = len(x)
    tr_prop = 0.7
    tr_size = int(n * tr_prop)

    # SparseGP and ExactGP must be run in full dataset with no reshuffling
    if args.loss in [
            'ExactGP', 'SparseGP', 'SparseGP2',
            'EigenSparseCVTGP', 'FullSparseCVTGP' #there are other versions that we did not include in the paper
            ]:                                    #these are for future work.
        print('{} loss does not accept minibatching!!!'.format(args.loss))
        batch_size=tr_size
        tr_shuffle=False # Don't want to reshuffle
        print('\t executing with batch_size = data_size = {} '.format(batch_size))
    else:
        batch_size=args.batch_size
        tr_shuffle=True

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    for fold in tqdm(range(args.cv_folds), position=0, leave=True):

        try:

            STOP = 0

            # Data set-up
            # val_test_batch is false for accurate ppl and gap calculations
            # otherwise, we approximate it with batches
            tr_dataloader, val_dataloader, te_dataloader, _ = prepare_fold(
                x, y, batch_size, fold, folds,
                args.device, dtype, dtypes, args.preprocess,
                tr_prop=tr_prop,
                tr_shuffle=tr_shuffle,
                val_test_batch=False
                )

            best_model = None
            if args.loss == 'SVGP':
                """
                This is standard SVGP - Hensman et al (init by kmeans)
                """
                model = SVGP(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    dtype=dtype,
                    inducing_point_size=args.ip_size
                ).to(args.device)

            elif args.loss == 'PPGPR':
                """
                PPGPR as described in the paper (init by kmeans)
                """
                model = PPGPR(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    dtype=dtype,
                    inducing_point_size=args.ip_size
                ).to(args.device)

            elif args.loss == 'SparseGP':
                """
                Titsias's VFE model (init by kmeans)
                """
                model = SparseGP(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    dtype=dtype,
                    inducing_point_size=args.ip_size
                ).to(args.device)

            elif args.loss == 'SparseGP2':
                """
                Another implementation of VFE - this time with torch
                """
                model = SparseGP2(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    dtype=dtype,
                    inducing_point_size=args.ip_size
                    ).to(args.device)

            elif args.loss == 'ExactGP':
                model = ExactGP(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    dtype=dtype,
                    ).to(args.device)

            elif args.loss == 'CVTGP':
                """
                Original CVTGP with SGD (init by kmeans)
                """
                model = CVTGP(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    inducing_point_size=args.ip_size,
                    dtype=dtype,
                    ).to(args.device)

            elif args.loss == 'RandomCVTGP':
                """
                Original CVTGP with SGD (init by randn)
                """
                model = RandomCVTGP(
                    kernel=args.kernel,
                    data=tr_dataloader.dataset.__getds__(False),
                    inducing_point_size=args.ip_size,
                    dtype=dtype,
                    ).to(args.device)

            parameters = [p for p in model.named_parameters()]
            if args.fh:
                parameters = [
                    (n,p) for (n,p) in parameters if (
                        'length' not in n
                        ) and (
                            'scale' not in n
                            )
                            ]
            elif args.fi:
                parameters = [
                    (n,p) for (n,p) in parameters if (
                        'x_c' not in n
                        ) and (
                            'y_c' not in n
                            )
                            ]
            else:
                parameters = [(n,p) for (n,p) in parameters]

            parameters = [p[-1] for p in parameters]
            optimizer = optim.Adam(
                parameters,
                lr=args.lr, weight_decay=args.wd
                )

            epoch_results = defaultdict(list)
            gap_results = defaultdict(list)
            if args.cv_folds == 1:
                fold = 'X'

            for epoch in tqdm(range(args.epochs), position=0, leave=True):

                try:
                    tr_loss, tr_score, tr_ppll = train_one_epoch(
                        model,
                        optimizer,
                        tr_dataloader,
                        METRIC
                        )

                    val_elbo, val_score, val_ppll = evaluate_model(
                        model, val_dataloader, METRIC,
                        )

                except (
                        ValueError, ZeroDivisionError, TypeError, RuntimeError
                        ):
                    #if there are any kernel error float precision etc.. 
                    #get best model back and keep training
                    model, optimizer = get_best_model_back(
                        model, best_model, optimizer,
                        args.fh, args.fi, args.lr, args.wd
                        )

                epoch_results = cache_epoch_results(
                    epoch_results, tr_loss, val_elbo,
                    tr_score, tr_ppll, val_score, val_ppll
                    )

                best_model = get_best_model(
                    best_model, model, epoch_results, METRIC,
                    )

                if epoch % args.check_stop == 0 and epoch != 0:
                    STOP = check_early_stop(
                        STOP, epoch, args.check_stop,
                        epoch_results, METRIC
                        )
                    #if you want to visualize coresets
                    # self = best_model
                    # plt.scatter(
                    #     self.x_c.detach().cpu()[:,0],
                    #     self.x_c.detach().cpu()[:,1],
                    #     # self.y_c.detach().cpu(),
                    #     c=nn.Softplus()(self.beta).detach().cpu()
                    #     )
                    # plt.show()
                    # self = best_model
                    # plt.scatter(
                    #     self.x_c.detach().cpu()[:,0],
                    #     self.x_c.detach().cpu()[:,1],
                    #     # self.y_c.detach().cpu(),
                    #     c=self.y_c.detach().cpu()
                    #     )
                    # plt.show()                    
                    try:
                        inference_gap = best_model.posterior_inference_gap(
                                val_dataloader.dataset.__getds__()[0]
                                )
                        gap_results['inference_gap_js'].append(
                            inference_gap[0]
                            )
                        gap_results['inference_gap_klpq'].append(
                            inference_gap[1]
                            )
                        gap_results['inference_gap_klqp'].append(
                            inference_gap[2]
                            )
                        gap_results['mu_rmse'].append(
                            inference_gap[3]
                            )
                        gap_results['cov_rmse'].append(
                            inference_gap[4]
                            )
                        gap_results['learning_gap'].append(
                            best_model.train().hyperparameter_learning_gap()
                            )
                    except:
                        pass

                elif STOP == args.early_stop:
                    print('Early stopping...')
                    break

            save_epoch_stats(
                model,
                epoch_results,
                gap_results,
                fold,
                args,
                METRIC,
                warm_up=0
                )

            fold_results = cache_fold_results(
                fold_results, best_model, te_dataloader,
                fold, METRIC
                )

        except KeyboardInterrupt:

            print('\nKeyboardInterrupt. Saving the results.')
            save_epoch_stats(
                model,
                epoch_results,
                gap_results,
                fold,
                args,
                METRIC,
                warm_up=0
                )

            fold_results = cache_fold_results(
                fold_results, best_model, te_dataloader,
                fold, METRIC
                )

        save_fold_stats(
            fold_results,
            best_model,
            fold,
            args,
            METRIC,
            )
