"""
Thin wrapper around tlp_train_mc_dropout for training Deep Ensemble seeds.

This ensures all models use the SAME AttentionModule class (with dropout layers),
so pickle.load works correctly in the ensemble eval script.

Usage:
  python3 tlp_train_seed.py \
    --dataset tlp_dataset_platinum_8272_2308_train_and_val.pkl \
    --save_folder runs/tlp_baseline_2308_seed3 \
    --n_epoch 20 \
    --cuda cuda:0 \
    --seed 3 \
    --dropout 0.0
"""
import sys
import os
import argparse
import pickle

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module (not specific names, because 'args' doesn't exist at
# module level in the old tlp_train_mc_dropout.py)
import tlp_train_mc_dropout as tlp_mc

# Build args namespace and inject it into tlp_train_mc_dropout's namespace
# so that AttentionModule.__init__ can access args.fea_size etc.
parser = argparse.ArgumentParser(description="Train TLP with configurable seed")
parser.add_argument("--save_folder", type=str, default='runs/tlp_baseline_2308_seed3')
parser.add_argument("--cuda", type=str, default='cuda:0')
parser.add_argument("--dataset", type=str,
                    default='tlp_dataset_platinum_8272_2308_train_and_val.pkl')
parser.add_argument("--lr", type=float, default=7e-4)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--rank_mse", type=str, default='rank')
parser.add_argument("--optimizer", type=str, default='default')
parser.add_argument("--attention_head", type=int, default=8)
parser.add_argument("--attention_class", type=str, default='default')
parser.add_argument("--step_size", type=int, default=25)
parser.add_argument("--fea_size", type=int, default=22)
parser.add_argument("--res_block_cnt", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--self_sup_model", type=str, default='')
parser.add_argument("--data_cnt", type=int, default=-1)
parser.add_argument("--train_size_per_gpu", type=int, default=1024)
parser.add_argument("--val_size_per_gpu", type=int, default=1024)
parser.add_argument("--n_epoch", type=int, default=20)
parser.add_argument("--seed", type=int, default=3)
cli_args = parser.parse_args()

# Set hidden_dim and out_dim (normally done inside train() based on attention_class)
if cli_args.attention_class == 'default':
    cli_args.hidden_dim = [64, 128, 256, 256]
    cli_args.out_dim = [256, 128, 64, 1]
elif cli_args.attention_class == 'attention_512':
    cli_args.hidden_dim = [64, 128, 256, 512]
    cli_args.out_dim = [256, 128, 64, 1]
elif cli_args.attention_class == 'attention_768':
    cli_args.hidden_dim = [64, 256, 512, 768]
    cli_args.out_dim = [512, 256, 128, 1]
elif cli_args.attention_class == 'attention_1024':
    cli_args.hidden_dim = [64, 256, 512, 1024]
    cli_args.out_dim = [512, 256, 128, 1]
else:
    cli_args.hidden_dim = [64, 128, 256, 256]
    cli_args.out_dim = [256, 128, 64, 1]

# CRITICAL: Inject args into tlp_train_mc_dropout's global namespace
# The model classes (AttentionModule etc.) reference 'args' as a global
tlp_mc.args = cli_args

# Set seed (overrides the set_seed(0) that ran on import)
tlp_mc.set_seed(cli_args.seed)

print(f"Training with seed={cli_args.seed}, dropout={cli_args.dropout}")
print(cli_args)

if not os.path.exists(cli_args.save_folder):
    os.makedirs(cli_args.save_folder, exist_ok=True)

print('load data...')
with open(cli_args.dataset, 'rb') as f:
    datasets_global = pickle.load(f)
print('load pkl done.')
datas = tlp_mc.load_datas(datasets_global)
print('create dataloader done.')
del datasets_global
print('load data done.')
tlp_mc.train(*datas, device=cli_args.cuda)
