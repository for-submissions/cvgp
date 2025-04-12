
from copy import deepcopy
import os

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
# plt.style.use('seaborn-paper')
# sns.set_palette("Dark2")
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams.update({
#     'font.size': 8,
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })

# from torch.distributions.normal import Normal
from gpytorch import add_jitter

class MultivariateNormal_:
    def __init__(self, loc, cov):
        self.mean = loc
        self.cov = cov
        self.k = self.mean.size(0)

    def log_prob(self, y):
        diff = y - self.mean
        cov_logdet = torch.log(torch.det(self.cov) + 1e-20)
        cov_inv = torch.linalg.inv(self.cov)
        exponent = -0.5 * torch.sum(diff @ cov_inv * diff)
        log_norm_const = -0.5 * (self.k * torch.log(torch.tensor(2 * torch.pi)) + cov_logdet)
        log_pdf = exponent + log_norm_const
        return log_pdf

def get_best_model_back(model, best_model, optimizer, fh, fi, lr, wd):
    #if there are nans due to weird landscape and optimization (momentum etc.)
    #go back to best model and re-init optimization.
    #applies for anymodel (cvtgp and baselines)
    # if (
    #         model.covar_module.raw_outputscale.isnan()
    #         ) or (
    #             model.covar_module.base_kernel.raw_lengthscale.isnan()
    #             ):
    model = deepcopy(best_model)
    parameters = [p for p in model.named_parameters()]
    if fh:
        parameters = [
            (n,p) for (n,p) in parameters if (
                'length' not in n
                ) and (
                    'scale' not in n
                    )
                    ]
    elif fi:
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
        lr=lr, weight_decay=wd
        )

    return model.train(), optimizer

def cholesky(matrix, initial_jitter=1e-5):
    initial_nan = torch.isnan(matrix).sum() \
        == matrix.shape[0] * matrix.shape[1]
    matrix = 0.5 * (torch.triu(matrix) + torch.tril(matrix).T)
    matrix = matrix.T + matrix - torch.eye(
        matrix.size(0)
        ).to(matrix.device) * torch.diag(matrix)
    jitter = initial_jitter
    chol_nan = torch.isnan(torch.linalg.cholesky_ex(matrix)[0].sum())
    matrix_ = matrix
    while chol_nan and ~initial_nan:
        matrix_ = add_jitter(matrix, jitter)
        jitter *= 1.1
        chol_nan = torch.isnan(torch.linalg.cholesky_ex(matrix_)[0].sum())
    u = torch.linalg.cholesky_ex(matrix_)[0]
    return u, jitter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device, dtype=torch.double):

        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(y, dtype=dtype),
            ] for x, y in zip(x, y)
        ]

        self.device = device
        self._cache = dict()

        self.input_size_ = x.shape[0]
        self.input_dim_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.device:
                self._cache[index][0] = self._cache[
                    index][0].to(self.device)

                self._cache[index][1] = self._cache[
                    index][1].to(self.device)

        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_dim(self):

        return self.input_dim_

    def input_size(self):

        return self.input_size_

    def __getds__(self, numpy=True):
        x = torch.cat([ds[0].view(1,-1) for ds in self.ds], axis=0)
        y = torch.stack([ds[1] for ds in self.ds])
        if numpy:
            x = to_np(x)
            y = to_np(y)
        return x, y

def one_hot_encode(dataframe, column):
    categorical = pd.get_dummies(dataframe[column], prefix=column)
    dataframe = dataframe.drop(column, axis=1)
    return pd.concat([dataframe, categorical], axis=1, sort=False)

def to_np(tensor):
    return torch.detach(tensor).cpu().numpy()

def transform(x, stats):
    loc, scale = stats
    return (x - loc) / scale

def inverse_transform(x, stats):
    if torch.is_tensor(x):
        x = to_np(x)
    loc, scale = stats
    return x * scale + loc

def rmse(y_pred, y):
    score = np.mean((y_pred - y)**2)**0.5
    return score

def prepare_fold(
        x, y, batch_size, fold, folds, device, tensor_dtype, dtypes, preprocess,
        tr_prop=0.7,
        tr_shuffle=True,
        val_test_batch=False
        ):

    print('\nPreparing fold : {}'.format(fold))

    if folds.sum() != 0:

        x_split = x[folds != fold]
        y_split = y[folds != fold]

        tr_size = int(x_split.shape[0] * tr_prop)

        x_tr, x_val = x_split[:tr_size], x_split[tr_size:]
        y_tr, y_val = y_split[:tr_size], y_split[tr_size:]

        x_te = x[folds == fold]
        y_te = y[folds == fold]

    else:

        tr_size = int(x.shape[0] * tr_prop)

        x_tr, x_val = x[:tr_size], x[tr_size:]
        y_tr, y_val = y[:tr_size], y[tr_size:]

        x_te, y_te = x_val, y_val

    loc = np.asarray(
        [
        x_tr[
            :,i
        ].mean(0) if dtypes[i]!='uint8' and preprocess else 0 for i in range(
                len(
                    dtypes
                    )
                )
                ]
                )

    scale = np.asarray(
        [
        1e-10 + x_tr[
            :,i
        ].std(0) if dtypes[i]!='uint8' and preprocess else 1 for i in range(
                len(
                    dtypes
                    )
                )
                ]
                )
    stats = np.concatenate([loc.reshape(1,-1), scale.reshape(1,-1)], 0)
    x_tr, x_val, x_te = [
        transform(x, stats) for x in [x_tr, x_val, x_te]
        ]

    train_data = Dataset(
        x_tr, y_tr, device, tensor_dtype
    )
    valid_data = Dataset(
        x_val, y_val, device, tensor_dtype
    )
    test_data = Dataset(
        x_te, y_te, device, tensor_dtype
    )

    tr_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=tr_shuffle
    )
    
    if val_test_batch:
        valid_batch_size = batch_size
        test_batch_size = batch_size
    else:
        valid_batch_size = x_val.shape[0]
        test_batch_size = x_te.shape[0]
    
    val_dataloader = DataLoader(
        valid_data, batch_size=valid_batch_size, shuffle=False
    )
    te_dataloader = DataLoader(
        test_data, batch_size=test_batch_size, shuffle=False
    )

    return tr_dataloader, val_dataloader, te_dataloader, stats

def train_one_epoch(model, optimizer, dataloader, metric):

    #define cache
    loss_list = []
    y_pred_list = []
    y_list = []
    predictive_ll_list = []
    # Model in train mode
    model.train()
    if model.__class__.__name__ in ['SVGP', 'PPGPR', 'SparseGP']:
        # Also likelihood
        model.likelihood.train()

    for x_tr, y_tr in dataloader:
        optimizer.zero_grad()
        # Compute loss
        if model.__class__.__name__ == 'SVGP':
            # Gpytorch Model output
            mvn = model(x_tr)
            # Associated loss
            loss = model.loss_f(
                        model.likelihood,
                        model,
                        num_data=x_tr.shape[0]
                    )(mvn, y_tr)
            # Keep model predictions only
            mvn = model.likelihood(mvn)
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        elif model.__class__.__name__ == 'PPGPR':
            # Gpytorch Model output
            mvn = model(x_tr)
            # Associated loss
            loss = model.loss_f(
                        model.likelihood,
                        model,
                        num_data=x_tr.shape[0]
                    )(mvn, y_tr)
            # Keep model predictions only
            mvn = model.likelihood(mvn)
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        elif model.__class__.__name__ == 'SparseGP':
            # Gpytorch Model output
            mvn = model(x_tr)
            # Associated loss
            loss = model.loss_f(
                        model.likelihood,
                        model,
                    )(mvn, y_tr)
            # Keep model predictions only
            mvn = model.likelihood(mvn)
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        elif model.__class__.__name__ == 'SparseGP2':
            loss, mvn = model(x_tr, y_tr)
            # Keep model predictions only
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        elif model.__class__.__name__ == 'ExactGP':
            loss, mvn = model(x_tr, y_tr)
            # Keep model predictions only
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        elif model.__class__.__name__ == 'CVTGP':
            loss, mvn = model(x_tr, y_tr)
            # Keep model predictions only
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        elif model.__class__.__name__ == 'RandomCVTGP':
            loss, mvn = model(x_tr, y_tr)
            # Keep model predictions only
            y_pred = mvn.mean.detach().cpu()
            predictive_ll = mvn.log_prob(y_tr).detach().cpu()
            predictive_ll = predictive_ll / y_tr.size(0)

        # Optimize
        (-loss).backward()            
        optimizer.step()

        loss_list.append(loss.item())
        y_pred_list.append(y_pred)
        y_list.append(y_tr)
        predictive_ll_list.append(predictive_ll)
    #kind of a biased estimate but it is ok.
    predictive_ll = torch.stack(predictive_ll_list).mean().item()
    y_pred = to_np(torch.cat(y_pred_list, 0))
    y_tr = to_np(torch.cat(y_list, 0))
    tr_score = rmse(y_pred, y_tr)
    tr_loss = np.mean(loss_list)

    return tr_loss, tr_score, predictive_ll

def evaluate_model(
        model, dataloader, metric,
        ):

    # Model into eval mode
    model.eval()
    if model.__class__.__name__ in ['SVGP', 'PPGPR', 'SparseGP']:
        # Also likelihood
        model.likelihood.eval()
    with torch.no_grad():

        for (x,y) in dataloader:

            loss_list = []
            y_pred_list = []
            y_list = []
            predictive_ll_list = []
            if model.__class__.__name__ == 'SVGP':
                # Gpytorch Model output
                mvn = model(x)
                # Associated loss
                loss = model.loss_f(
                            model.likelihood,
                            model,
                            num_data=x.shape[0]
                        )(mvn, y)
                mvn = model.likelihood(mvn)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll = mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)

            elif model.__class__.__name__ == 'PPGPR':
                # Gpytorch Models' output
                mvn = model(x)
                # Associated loss
                loss = model.loss_f(
                            model.likelihood,
                            model,
                            num_data=x.shape[0]
                        )(mvn, y)
                mvn = model.likelihood(mvn)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll = mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)

            elif model.__class__.__name__ == 'SparseGP':
                # Gpytorch Model output
                mvn = model(x)
                # Associated loss
                loss = model.loss_f(
                            model.likelihood,
                            model,
                        )(mvn, y)
                mvn = model.likelihood(mvn)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll =mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)

            elif model.__class__.__name__ == 'SparseGP2':
                loss, mvn = model(x, y)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll = mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)

            elif model.__class__.__name__ == 'ExactGP':
                loss, mvn = model(x, y)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll = mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)

            elif model.__class__.__name__ == 'CVTGP':
                loss, mvn = model(x, y)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll = mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)

            elif model.__class__.__name__ == 'RandomCVTGP':
                loss, mvn = model(x, y)
                # Keep model predictions only
                y_pred = mvn.mean.detach().cpu()
                predictive_ll = mvn.log_prob(y).detach().cpu()
                predictive_ll = predictive_ll / y.size(0)


            loss_list.append(loss)
            y_pred_list.append(y_pred)
            y_list.append(y)
            predictive_ll_list.append(predictive_ll)

        predictive_ll = torch.stack(predictive_ll_list).mean().item()
        y_pred = to_np(torch.cat(y_pred_list, 0))
        y = to_np(torch.cat(y_list, 0))
        score = rmse(y_pred, y)
        loss = torch.stack(loss_list).mean().item()

    return loss, score, predictive_ll

def get_best_model(best_model, model, epoch_results, METRIC):
    if METRIC == 'RMSE':
        if epoch_results['Valid RMSE'][-1] <= epoch_results['Valid Best RMSE'][-1]:
            best_model = deepcopy(model)
    elif METRIC == 'PPLL':
        if epoch_results['Valid PPLL'][-1] >= epoch_results['Valid Best PPLL'][-1]:
            best_model = deepcopy(model)
    else:
        if epoch_results['Valid Loss'][-1] >= epoch_results['Valid Best Loss'][-1]:
            best_model = deepcopy(model)
    try:
        best_model = best_model.eval()
    except:
        best_model = model.eval()
    return best_model

def check_early_stop(
        stop, epoch, check_early_stop, epoch_results, METRIC
        ):

    print('\n\nValidating\
        \nEpoch Valid Loss : {:.2E} | Epoch Valid RMSE : {:.3f} \
        \nValid Rolling Mean Loss : {:.3f} | Valid Mean RMSE : {:.3f} |\
        \nValid Best RMSE : {:.3f}\
        \nValid Best Loss : {:.3f}\
        \nCaching Best : {} \
        \n\n'
        .format(
            epoch_results['Valid Loss'][-1],
            epoch_results['Valid RMSE'][-1],
            epoch_results['Valid Mean Loss'][-1],
            epoch_results['Valid Mean RMSE'][-1],
            epoch_results['Valid Best RMSE'][-1],
            epoch_results['Valid Best Loss'][-1],
            METRIC
            ),
        end="\r"
        )

    if METRIC == 'RMSE':
        if epoch_results[
                'Valid Best RMSE'
                ][-1] >= epoch_results[
                    'Valid Best RMSE'
                    ][-check_early_stop]:
            stop += 1
        else:
            stop = 0
    else:
        if epoch_results[
                'Valid Best Loss'
                ][-1] <= epoch_results[
                    'Valid Best Loss'
                    ][-check_early_stop]:
            stop += 1
        else:
            stop = 0

    print('STOP : {}'.format(stop))

    return stop

def cache_epoch_results(
        epoch_results, tr_loss, val_loss,
        tr_score, tr_ppll, val_score, val_ppll
        ):

    epoch_results['Train Loss'].append(tr_loss)
    epoch_results['Valid Loss'].append(val_loss)
    epoch_results['Train RMSE'].append(tr_score)
    epoch_results['Valid RMSE'].append(val_score)
    epoch_results['Train PPLL'].append(tr_ppll)
    epoch_results['Valid PPLL'].append(val_ppll)


    epoch_results['Train Best Loss'].append(max(epoch_results['Train Loss']))
    epoch_results['Train Best RMSE'].append(min(epoch_results['Train RMSE']))
    epoch_results['Train Best PPLL'].append(max(epoch_results['Train PPLL']))
    epoch_results['Valid Best Loss'].append(max(epoch_results['Valid Loss']))
    epoch_results['Valid Best RMSE'].append(min(epoch_results['Valid RMSE']))
    epoch_results['Valid Best PPLL'].append(max(epoch_results['Valid PPLL']))

    epoch_results['Train Mean Loss'].append(
        np.mean(epoch_results['Train Loss'])
        )
    epoch_results['Train Mean RMSE'].append(
        np.mean(epoch_results['Train RMSE'])
        )
    epoch_results['Train Mean PPLL'].append(
        np.mean(epoch_results['Train PPLL'])
        )

    epoch_results['Valid Mean Loss'].append(
        np.mean(epoch_results['Valid Loss'])
        )
    epoch_results['Valid Mean RMSE'].append(
        np.mean(epoch_results['Valid RMSE'])
        )
    epoch_results['Valid Mean PPLL'].append(
        np.mean(epoch_results['Valid PPLL'])
        )

    return epoch_results

def save_epoch_stats(
        model,
        epoch_results, gap_results, fold,
        args, METRIC,
        warm_up=0
        ):

    FLAGS = ', '.join(
        [
            str(y) + ' ' + str(x) for (y,x) in vars(args).items() if y not in [
                'dataset',
                'loss',
                ]
            ]
        )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax[0].plot(
        epoch_results['Train Loss'][warm_up:], color='b', label='Train Loss'
        )
    ax[0].plot(
        epoch_results['Valid Loss'][warm_up:], color='r', label='Valid Loss'
        )

    if model.__class__.__name__ == 'marginal':
        label = r'$\mathcal{L} (Log-likelihood)$'
    elif model.__class__.__name__ == 'em' or model.__class__.__name__ == 'GeneralizedEM':
        label =\
    r'$Q(\theta, \theta^\prime \mid \mathbf{X}^\prime, \mathbf{y}^\prime)$'
    else:
        label=r'$\mathcal{L}$'
    ax[0].set_ylabel(
        label,
        size=35
        )
    ax[0].set_xlabel(
        'Epochs',
        size=35
        )
    ax[0].tick_params(axis='both', labelsize=25)
    # ax[0].tick_params(axis='both', which='minor',labelsize=20)

    color = ['b', 'r']
    i = 0
    for (key, value) in epoch_results.items():
        if METRIC in key:
            if ('Best' not in key) and ('Mean' not in key):
                ax[1].plot(value[warm_up:], color=color[i], label=key)
                i += 1

    ax[1].set_ylabel('{}'.format(METRIC),
                     size=30
                     )
    ax[1].set_xlabel(
        'Epochs',
        size=35
        )
    ax[1].tick_params(axis='both', labelsize=25)
    # ax[1].tick_params(axis='both', which='minor',labelsize=20)

    plt.tight_layout()

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.27,1), prop={'size': 30})

    os.makedirs('./fold_figures', exist_ok=True)
    plt.savefig("./fold_figures/{}_{}_fold_{}_figs_({}).pdf".format(
        args.loss,
        args.dataset,
        fold,
        FLAGS
        ),
        bbox_inches='tight'
        )

    epoch_results = pd.DataFrame(epoch_results)
    os.makedirs('./epoch_results', exist_ok=True)
    epoch_results.to_csv(
        './epoch_results/{}_{}_fold_{}_epoch_res_({}).csv'.format(
            args.loss,
            args.dataset,
            fold,
            FLAGS
            )
        )
    gap_results = pd.DataFrame(gap_results)
    if not gap_results.empty:
        os.makedirs('./gap_results', exist_ok=True)
        gap_results.to_csv(
            './gap_results/{}_{}_fold_{}_gap_results_({}).csv'.format(
                args.loss,
                args.dataset,
                fold,
                FLAGS
                )
            )

def cache_fold_results(
        fold_results, best_model, te_dataloader, fold, METRIC
        ):

    loss, metric, ppll = evaluate_model(
        best_model.eval(),
        te_dataloader,
        METRIC
        )

    fold_results['Fold: {}'.format(fold)]['RMSE'].append(metric)
    fold_results['Fold: {}'.format(fold)]['Loss'].append(loss)
    fold_results['Fold: {}'.format(fold)]['PPLL'].append(ppll)

    print('\n\n'+'-'*80)
    print(
        'Testing Fold : {} | LOSS : {:.2E} | Test Score : {:.3f} | \n'
        .format(fold, loss, metric), end="\r"
          )
    print('-'*80)

    return fold_results

def save_fold_stats(fold_results, best_model, fold, args, METRIC):

    FLAGS = ', '.join(
        [
            str(y) + ' ' + str(x) for (y,x) in vars(args).items() if y not in [
                'dataset',
                'loss',
                ]
            ]
        )

    if len(fold_results) != 0:
        print('\n' + '-'*26 + 'Results' + '-'*27)
        print('Mean Fold Score : {}'.format(
            np.mean(
                fold_results[
                    'Fold: {}'.format(fold)
                    ][METRIC]
                )
            )
            )
        print('Standard Deviation : {}'.format(
            np.std(
                fold_results[
                    'Fold: {}'.format(fold)
                    ][METRIC]
                )
            )
            )
        print('Fold Scores : {}'.format(
            fold_results[
                'Fold: {}'.format(fold)
                ][METRIC]
            )
            )
        print('-'*60 + '\n')

    os.makedirs('./model_checkpoints', exist_ok=True)

    try:
        torch.save(
            best_model,
            './model_checkpoints/{}_{}_fold_{}_({}).pth'.format(
                args.loss,
                args.dataset,
                fold,
                FLAGS
                )
            )
    except:
        print('Unable to save model checkpoint for {}_{}_fold_{}_({}).pth'.format(
                args.loss,
                args.dataset,
                fold,
                FLAGS
            )
        )
    fold_results = pd.DataFrame(fold_results)
    for key in fold_results.keys():
        fold_results[key] = [
            _[0] for _ in fold_results[key]
        ]
    os.makedirs('./fold_results', exist_ok=True)
    fold_results.to_csv(
        './fold_results/{}_{}_fold_results_({}).csv'.format(
            args.loss,
            args.dataset,
            FLAGS
            )
        )
