import utils
import os
import sys
import time
import glob
import argparse
import logging
import train_all_search_gradient
import train_all

parser = argparse.ArgumentParser()

parser.add_argument('--epochs_pre_prune', type=int, default=0, help='epochs pre pruning')
parser.add_argument('--steps_accum', type=int, default=0, help='number of steps to accumulate gradients')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--exponent', type=int, default=3, help='sparsity')
parser.add_argument('--epochs_s', type=int, default=50, help='epochs')
parser.add_argument('--layers', type=int, default=8, help='layers')


args = parser.parse_args()

args.save = 'ALL-accum-' + str(args.steps_accum) + '-pre-' + str(args.epochs_pre_prune)+'-'+time.strftime("%Y%m%d-%H%M%S")
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)


def main():

    args_ts = train_all_search_gradient.Args(args.gpu, args.unrolled, args.epochs_pre_prune, args.steps_accum, args.save, args.exponent, args.epochs_s, args.layers)
    genotype = train_all_search_gradient.search_phase(logging, args_ts)
    logging.info(' *** end of search ***')
    args_t = train_all.Args(args.gpu, args.auxiliary, args.cutout, args.save)
    logging.info('*** start train ***')
    train_all.train_phase(genotype, logging, args_t)


if __name__ == '__main__':
    main()