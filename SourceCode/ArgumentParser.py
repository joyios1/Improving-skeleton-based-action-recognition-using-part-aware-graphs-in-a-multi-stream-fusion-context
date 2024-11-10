
import argparse
import yaml
import os

from texttable import Texttable


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    if args.phase == 'test' or args.weights is not None:
        return
    args = vars(args)
    keys = sorted(args.keys())
    # keys = args.keys()
    t = Texttable()
    t.set_precision(5)
    t.header(["Parameter", "Value"])
    args_list = [[k.replace("_", " "), args[k]] for k in keys]
    for arg in args_list:
        t.add_row(arg)
    print(t.draw())
    if not os.path.exists(args['work_dir']):
        os.makedirs(args['work_dir'])
    with open('{}/log.txt'.format(args['work_dir']), 'w') as f:
        print(t.draw(), file=f)


def load_arguments_from_yaml(parser):
    parsed_args = parser.parse_args()
    with open(parsed_args.config, 'r') as f:
        yaml_arguments = yaml.full_load(f)
    parser_keys = vars(parsed_args).keys()
    for yaml_argument_key in yaml_arguments.keys():
        if yaml_argument_key not in parser_keys:
            print('WRONG ARGUMENT: {}'.format(yaml_argument_key))
            assert (yaml_argument_key in parser_keys)
    parser.set_defaults(**yaml_arguments)

    return parser.parse_args()


def get_parser():
    parser = argparse.ArgumentParser(description="Spatial Temporal Graph Convolution Network")

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--work-dir', default='./working_directory/temp', help='The work folder for storing results.')
    parser.add_argument('--config', default='./config/nturgbd-cross-subject/default.yaml', required=True, help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visualize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=1, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-top-k', type=int, default=[1, 5], nargs='+', metavar='', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default=None, help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=0, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--val-feeder-args', default=dict(), help='the arguments of data loader for validating')
    parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--graph', default=dict(), help='the arguments of model')
    parser.add_argument('--model-args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='The epoch where optimizer reduce the learning rate.')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='The indexes of GPUs for training or testing.')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=90, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='Decay rate for learning rate')
    parser.add_argument('--lr-ratio', type=float, default=0.0001, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser
