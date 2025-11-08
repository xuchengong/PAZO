import torch
import numpy as np
import random
import argparse

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def none_or_str(value):
    if value.lower() == 'none':
        return None
    return value


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_from',
                        help='from which folder under log to continue training',
                        type=none_or_str,
                        )
    parser.add_argument('--private_bs',
                        help='batchsize of g_private',
                        type=int,
                        default=128,
                        )
    parser.add_argument('--num_test_per_epoch',
                        help='number of tests each epoch',
                        type=int,
                        default=4,
                        )
    parser.add_argument('--public_bs',
                        help='batchsize of g_public',
                        type=int,
                        default=50)
    parser.add_argument('--num_candidate',
                        help='how many candidates of public gradients that is used to construct G',
                        type=int,
                        default=0)
    parser.add_argument('--time',
                        help='time when the experiment is run',
                        type=str)
    parser.add_argument('--public_size',
                        help='number of train data points as public data',
                        type=int,
                        default=500,)
    parser.add_argument('--dataset',
                        help='name of the dataset',
                        type=str,
                        default='cifar10')
    parser.add_argument('--pretrain',
                        help='which weights to start from',
                        type=str,)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=0.01)
    parser.add_argument('--num_directions',
                        help='number of random directions for zeroth order optimization',
                        type=int,
                        default=1)
    parser.add_argument('--num_microbatches',
                        help='how many microbatches in one mini-batch, only for dp methods',
                        type=int,
                        default=8)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
    parser.add_argument('--eval_every_epoch',
                        type=int,
                        default=1)
    parser.add_argument('--eval_every_iter',
                        type=int,
                        default=200)
    parser.add_argument('--seed',
                        help='numpy seed',
                        type=int,
                        default=0)
    parser.add_argument('--sigma',
                        help='sigma parameter of gaussian mechanism (== noise_std / l2_bound)',
                        type=float,
                        default=1.0)
    parser.add_argument('--delta',
                        help='delta in the privacy parameters',
                        type=float,
                        default=1e-5)
    parser.add_argument('--clipping_bound',
                        help='gradient clipping for private training, or gradient clipping for zeroth-order methods',
                        type=float,
                        default=1e5,
                        )
    parser.add_argument('--momentum',
                        help='whether to use momentum',
                        type=int,
                        default=0)
    parser.add_argument('--epsilon_scale',
                        help='noise scale as residual gradient',
                        type=float,
                        default=0.01)
    parser.add_argument('--perturbation_scale',
                        help='for zeroth order optimization',
                        type=float,
                        default=0.001)
    parser.add_argument('--coefficient',
                        help='coefficients for linear combination',
                        type=float,
                        default=0.5)
    parser.add_argument('--augmented',
                        help='whether use augmentation on public data',
                        type=int,
                        default=1)
    parser.add_argument('--target_eps',
                        type=float,
                        default=3.0)
    parser.add_argument('--wd',
                        help='weight decay',
                        type=float,
                        default=1e-4)
    
    # GEP hyperparameters
    parser.add_argument('--clip0', default=5., type=float, help='clipping threshold for gradient embedding')
    parser.add_argument('--clip1', default=2., type=float, help='clipping threshold for residual gradients')
    parser.add_argument('--power_iter', default=1, type=int, help='number of power iterations')
    parser.add_argument('--num_groups', default=3, type=int, help='number of parameters groups')
    parser.add_argument('--num_bases', default=1000, type=int, help='dimension of anchor subspace')

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    setup_seed(parsed['seed'])
    return parsed
