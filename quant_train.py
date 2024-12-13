import argparse
import os
import random
import shutil
import time
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from bit_config import *
from utils import *
from pytorchcv.model_provider import get_model as ptcv_get_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--teacher-arch',
                    type=str,
                    default='resnet101',
                    help='teacher network used to do distillation')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--act-range-momentum',
                    type=float,
                    default=-1,
                    help='momentum of the activation range moving average, '
                         '-1 stands for using minimum of min and maximum of max')
parser.add_argument('--quant-mode',
                    type=str,
                    default='symmetric',
                    choices=['asymmetric', 'symmetric'],
                    help='quantization mode')
parser.add_argument('--save-path',
                    type=str,
                    default='checkpoints/imagenet/test/',
                    help='path to save the quantized model')
parser.add_argument('--data-percentage',
                    type=float,
                    default=1,
                    help='data percentage of training data')
parser.add_argument('--fix-BN',
                    action='store_true',
                    help='whether to fix BN statistics and fold BN during training')
parser.add_argument('--fix-BN-threshold',
                    type=int,
                    default=None,
                    help='when to start training with fixed and folded BN,'
                         'after the threshold iteration, the original fix-BN will be overwritten to be True')
parser.add_argument('--checkpoint-iter',
                    type=int,
                    default=-1,
                    help='the iteration that we save all the featuremap for analysis')
parser.add_argument('--evaluate-times',
                    type=int,
                    default=-1,
                    help='The number of evaluations during one epoch')
parser.add_argument('--quant-scheme',
                    type=str,
                    default='uniform4',
                    help='quantization bit configuration')
parser.add_argument('--resume-quantize',
                    action='store_true',
                    help='if True map the checkpoint to a quantized model,'
                         'otherwise map the checkpoint to an ordinary model and then quantize')
parser.add_argument('--act-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for activation percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--weight-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for weight percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--channel-wise',
                    action='store_false',
                    help='whether to use channel-wise quantizaiton or not')
parser.add_argument('--bias-bit',
                    type=int,
                    default=32,
                    help='quantizaiton bit-width for bias')
parser.add_argument('--distill-method',
                    type=str,
                    default='None',
                    help='you can choose None or KD_naive')
parser.add_argument('--distill-alpha',
                    type=float,
                    default=0.95,
                    help='how large is the ratio of normal loss and teacher loss')
parser.add_argument('--temperature',
                    type=float,
                    default=6,
                    help='how large is the temperature factor for distillation')
parser.add_argument('--fixed-point-quantization',
                    action='store_true',
                    help='whether to skip deployment-oriented operations and '
                         'use fixed-point rather than integer-only quantization')

best_acc1 = 0
quantize_arch_dict = {'resnet50': q_resnet50, 'resnet50b': q_resnet50,
                      'resnet18': q_resnet18, 'resnet101': q_resnet101,
                      'inceptionv3': q_inceptionv3,
                      'mobilenetv2_w1': q_mobilenetv2_w1}
args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

hook_counter = args.checkpoint_iter
hook_keys = []
hook_keys_counter = 0

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=args.save_path + 'log.log')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args)


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.arch == 'gpt':
        main_worker_gpt(args.gpu, ngpus_per_node, args)
    elif args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker_gpt(gpu, ngpus_per_node, args):
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import os
    import logging
    from torch.nn.parameter import Parameter
    from torch.optim import AdamW

    # Set up device
    args.gpu = gpu
    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save_path exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not hasattr(args, 'bias_bit') or args.bias_bit is None:
        args.bias_bit = 32

    quantize_bias = (args.bias_bit != 32)

    # -----------------------------
    # Data Loading and Tokenization
    # -----------------------------
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    vocab_size = tokenizer.n_vocab

    data_dir = "data.txt"
    with open(data_dir, 'r', encoding='utf-8') as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    train_batch_size = args.batch_size if args.batch_size is not None else 1
    eval_batch_size = 1
    context_length = 256
    train_split = 0.7
    n_data = len(data)
    train_data = data[:int(n_data * train_split)]
    eval_data = data[int(n_data * train_split):]

    class DataLoader:
        def __init__(self, tokens, batch_size, context_length):
            self.tokens = tokens
            self.batch_size = batch_size
            self.context_length = context_length
            self.current_position = 0

        def get_batch(self):
            b, c = self.batch_size, self.context_length
            start_pos = self.current_position
            end_pos = self.current_position + b * c + 1

            add_data = -1
            if end_pos > len(self.tokens):
                add_data = end_pos - len(self.tokens)
                end_pos = len(self.tokens)

            d = self.tokens[start_pos:end_pos]
            if add_data != -1:
                d = torch.cat([d, self.tokens[:add_data]])

            x = (d[:-1]).view(b, c)
            y = (d[1:]).view(b, c)

            self.current_position += b * c
            if self.current_position > len(self.tokens) - 1:
                self.current_position = 0

            return x.to(device), y.to(device)

    train_loader = DataLoader(train_data, train_batch_size, context_length)
    eval_loader = DataLoader(eval_data, eval_batch_size, context_length)

    # -----------------------------
    # Model Definitions (GPT & Q_GPT)
    # -----------------------------
    d_model = 512
    n_heads = 4
    n_layers = 3

    class PositionalEncoding(nn.Module):
        def __init__(self, context_length, d_model):
            super().__init__()
            pe = torch.zeros(context_length, d_model)
            position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            assert (n_heads * self.head_dim == d_model)
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.value = nn.Linear(d_model, d_model)
            self.fc_out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(0.2)

        def forward(self, inputs):
            B, seq_length, d_model = inputs.shape
            Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
            scores = scores.masked_fill(mask, float('-inf'))
            att_weights = torch.softmax(scores, dim=-1)
            att_output = torch.matmul(self.dropout(att_weights), V)
            att_output = att_output.permute(0, 2, 1, 3).contiguous().view(B, seq_length, d_model)
            out = self.fc_out(att_output)
            return out

    class GPTBlock(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.att = MultiHeadAttention(d_model, n_heads)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(0.2)
            self.fcn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

        def forward(self, logits):
            att_logits = self.att(self.ln1(logits))
            logits = logits + self.dropout(att_logits)
            fcn_logits = self.fcn(self.ln2(logits))
            logits = logits + self.dropout(fcn_logits)
            return logits

    class GPT(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.wte = nn.Embedding(vocab_size, d_model)
            self.wpe = PositionalEncoding(context_length, d_model)
            self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
            self.linear1 = nn.Linear(d_model, vocab_size)

        def forward(self, inputs, targets=None):
            logits = self.wte(inputs)
            logits = self.wpe(logits)
            for block in self.blocks:
                logits = block(logits)
            logits = self.linear1(logits)
            loss = None
            if targets is not None:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

        def generate(self, inputs, max_new_tokens):
            output = inputs.clone()
            for _ in range(max_new_tokens):
                current_seq_length = output.size(1)
                if current_seq_length > context_length:
                    inputs_trunc = output[:, -context_length:]
                else:
                    inputs_trunc = output
                logits, _ = self(inputs_trunc)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=1)
                idx_next = torch.multinomial(probs, num_samples=1)
                output = torch.cat([output, idx_next], dim=1)
            return [tokenizer.decode(o.tolist()) for o in output]

    # Q_GPT classes
    class Q_MultiHeadAttention(nn.Module):
        def __init__(self, dim, n_heads, bit_config):
            super().__init__()
            self.n_heads = n_heads
            self.dim = dim
            self.head_dim = dim // n_heads

            # Create the QuantLinear layers
            self.query = QuantLinear(weight_bit=bit_config['blocks.0.att.query'])
            self.key = QuantLinear(weight_bit=bit_config['blocks.0.att.key'])
            self.value = QuantLinear(weight_bit=bit_config['blocks.0.att.value'])
            self.fc_out = QuantLinear(weight_bit=bit_config['blocks.0.att.fc_out'])

            # Initialize them from normal Linear layers
            query_linear = nn.Linear(dim, dim)
            key_linear = nn.Linear(dim, dim)
            value_linear = nn.Linear(dim, dim)
            fc_linear = nn.Linear(dim, dim)

            self.query.set_param(query_linear)
            self.key.set_param(key_linear)
            self.value.set_param(value_linear)
            self.fc_out.set_param(fc_linear)

            self.quant_act = QuantAct(activation_bit=bit_config['blocks.0.att.quant_act'])
            self.quant_act_int32 = QuantAct(activation_bit=bit_config['blocks.0.att.quant_act_int32'])
            self.dropout = nn.Dropout(0.2)

        def set_param(self, original_att):
            # original_att is the MultiHeadAttention from the original GPTBlock
            # It has query, key, value, fc_out as nn.Linear layers.
            self.query.set_param(original_att.query)
            self.key.set_param(original_att.key)
            self.value.set_param(original_att.value)
            self.fc_out.set_param(original_att.fc_out)

        def forward(self, x):
            B, seq_len, _ = x.shape
            q, qs = self.quant_act(self.query(x))
            k, ks = self.quant_act(self.key(x))
            v, vs = self.quant_act(self.value(x))
            q = q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1,2)
            k = k.view(B, seq_len, self.n_heads, self.head_dim).transpose(1,2)
            v = v.view(B, seq_len, self.n_heads, self.head_dim).transpose(1,2)
            scores = torch.matmul(q, k.transpose(-1,-2)) / (self.head_dim**0.5)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))
            att_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(att_weights, v)
            out = out.transpose(1,2).contiguous().view(B, seq_len, self.dim)
            out, outs = self.quant_act_int32(self.fc_out(out, vs))
            return out

    class Q_FeedForward(nn.Module):
        def __init__(self, dim, bit_config):
            super().__init__()
            self.quant_act = QuantAct(activation_bit=bit_config['blocks.0.fcn.quant_act'])
            self.quant_act_int32 = QuantAct(activation_bit=bit_config['blocks.0.fcn.quant_act_int32'])

            self.linear1 = QuantLinear(weight_bit=bit_config['blocks.0.fcn.linear1'])
            self.linear2 = QuantLinear(weight_bit=bit_config['blocks.0.fcn.linear2'])

            # Temporary initialization, set_param will overwrite these
            l1 = nn.Linear(dim, 4*dim)
            l2 = nn.Linear(4*dim, dim)
            self.linear1.set_param(l1)
            self.linear2.set_param(l2)

            self.activation = nn.GELU()

        def set_param(self, original_fcn):
            # original_fcn is a nn.Sequential: [Linear(d_model,4*d_model), GELU, Linear(4*d_model,d_model)]
            # so original_fcn[0] and original_fcn[2] are the linear layers.
            self.linear1.set_param(original_fcn[0])
            self.linear2.set_param(original_fcn[2])

        def forward(self, x):
            x, xs = self.quant_act(self.linear1(x))
            x = self.activation(x)
            x, x2s = self.quant_act_int32(self.linear2(x, xs))
            return x

    class Q_GPTBlock(nn.Module):
        def __init__(self, d_model, n_heads, bit_config):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.att = Q_MultiHeadAttention(d_model, n_heads, bit_config)
            self.fcn = Q_FeedForward(d_model, bit_config)
            self.dropout = nn.Dropout(0.2)

        def set_param(self, original_block):
            # original_block is a GPTBlock from the original model
            # It has ln1, ln2, att, fcn
            # We need to set parameters for att and fcn from the original block.
            self.ln1.load_state_dict(original_block.ln1.state_dict())
            self.ln2.load_state_dict(original_block.ln2.state_dict())

            self.att.set_param(original_block.att)
            self.fcn.set_param(original_block.fcn)

        def forward(self, x):
            att_out = self.att(self.ln1(x))
            x = x + self.dropout(att_out)
            ff_out = self.fcn(self.ln2(x))
            x = x + self.dropout(ff_out)
            return x  # no act_scaling_factor returned

    class QuantEmbedding(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, weight_bit=8):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight_bit = weight_bit
            self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))
            nn.init.normal_(self.weight, mean=0, std=embedding_dim**-0.5)
            self.register_buffer('weight_scaling_factor', torch.zeros(1))

        def forward(self, x):
            w_min = self.weight.data.min()
            w_max = self.weight.data.max()
            scale = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max)
            w_int = SymmetricQuantFunction.apply(self.weight, self.weight_bit, scale)
            emb = F.embedding(x, w_int) * scale
            return emb

    class Q_GPT(nn.Module):
        def __init__(self, model, bit_config):
            super().__init__()
            d_model = model.d_model
            n_layers = model.n_layers
            n_heads = model.n_heads
            vocab_size = model.vocab_size

            # Quantized embedding
            self.wte = QuantEmbedding(vocab_size, d_model)
            # set_param if implemented, or just copy weights
            with torch.no_grad():
                self.wte.weight.copy_(model.wte.weight.data)

            # Positional encoding (not trainable, just quantize output)
            self.wpe = model.wpe
            # self.quant_act_wpe = QuantAct()

            # Quantized GPT Blocks
            self.blocks = nn.ModuleList([Q_GPTBlock(d_model, n_heads, bit_config=bit_config) for _ in range(n_layers)])
            # self.blocks = model.blocks
            self.n_layers = n_layers
            # Each Q_GPTBlock should have a set_param method
            # that sets parameters from the corresponding original block if needed.
            # self.linear1 = nn.Linear(d_model, vocab_size)

            # Quantize output activation and final linear
            self.quant_act_output = QuantAct()
            self.quant_output = QuantLinear()
            self.quant_output.set_param(model.linear1)

        def set_param(self, original_gpt):
            # original_gpt is the GPT model
            # set parameters for each block
            for i in range(self.n_layers):
                self.blocks[i].set_param(original_gpt.blocks[i])
            # wte and wpe are already set (wpe has no params)
            # quant_output is already set
            # If you'd like to handle layernorm, etc., do so here if needed.
            # If the original model had layernorm parameters for each block's ln1, ln2,
            # they've already been handled in Q_GPTBlock's set_param method.

        def forward(self, x, targets=None):
            # If you don't need act_scaling_factor at all, remove it here
            x = self.wte(x)
            x = self.wpe(x)

            # Now just forward through the blocks without act_scaling_factor
            for block in self.blocks:
                x = block(x)

            # # Same at the output
            x, act_scaling_factor = self.quant_act_output(x)
            B, T, C = x.shape
            x = x.view(B*T, C)
            x = self.quant_output(x, act_scaling_factor)

            loss = None
            if targets is not None:
                targets = targets.view(B*T)
                loss = F.cross_entropy(x, targets)
            return x, loss

    module_config = {
        'wte': 8,
        'blocks': 8,
        'blocks.0': 8,
        'blocks.0.ln1': 8,
        'blocks.0.ln2': 8,
        'blocks.0.att': 8,
        'blocks.0.att.query': 8,
        'blocks.0.att.key': 8,
        'blocks.0.att.value': 8,
        'blocks.0.att.fc_out': 8,
        'blocks.0.att.quant_act': 8,
        'blocks.0.att.quant_act_int32': 16,
        'blocks.0.att.dropout': 8,
        'blocks.0.fcn': 8,
        'blocks.0.fcn.quant_act': 8,
        'blocks.0.fcn.quant_act_int32': 16,
        'blocks.0.fcn.linear1': 8,
        'blocks.0.fcn.linear2': 8,
        'blocks.0.fcn.activation': 8,
        'blocks.0.dropout': 8,
        'blocks.1': 8,
        'blocks.1.ln1': 8,
        'blocks.1.ln2': 8,
        'blocks.1.att': 8,
        'blocks.1.att.query': 8,
        'blocks.1.att.key': 8,
        'blocks.1.att.value': 8,
        'blocks.1.att.fc_out': 8,
        'blocks.1.att.quant_act': 8,
        'blocks.1.att.quant_act_int32': 16,
        'blocks.1.att.dropout': 8,
        'blocks.1.fcn': 8,
        'blocks.1.fcn.quant_act': 8,
        'blocks.1.fcn.quant_act_int32': 16,
        'blocks.1.fcn.linear1': 8,
        'blocks.1.fcn.linear2': 8,
        'blocks.1.fcn.activation': 8,
        'blocks.1.dropout': 8,
        'blocks.2': 8,
        'blocks.2.ln1': 8,
        'blocks.2.ln2': 8,
        'blocks.2.att': 8,
        'blocks.2.att.query': 8,
        'blocks.2.att.key': 8,
        'blocks.2.att.value': 8,
        'blocks.2.att.fc_out': 8,
        'blocks.2.att.quant_act': 8,
        'blocks.2.att.quant_act_int32': 16,
        'blocks.2.att.dropout': 8,
        'blocks.2.fcn': 8,
        'blocks.2.fcn.quant_act': 8,
        'blocks.2.fcn.quant_act_int32': 16,
        'blocks.2.fcn.linear1': 8,
        'blocks.2.fcn.linear2': 8,
        'blocks.2.fcn.activation': 8,
        'blocks.2.dropout': 8,
        'quant_act_output': 8,
        'quant_output': 8
    }

    # -----------------------------
    # Create and Quantize Model
    # -----------------------------
    base_model = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    logging.info(base_model)

    for name, m in base_model.named_modules():
        logging.info(f"Old Module: {name}")

    model = Q_GPT(base_model, bit_config=module_config).to(device)
    model.set_param(base_model)

    for name, m in model.named_modules():
        logging.info(f"New Module: {name}")

    logging.info(model)

    logging.info(args)
    
    for name, m in model.named_modules():
        logging.info(f"Module: {name}")
        if name in module_config.keys():
            logging.info(f"Setting quantization parameters for {name}")
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', args.bias_bit)
            setattr(m, 'quantize_bias', quantize_bias)
            setattr(m, 'per_channel', args.channel_wise)
            setattr(m, 'act_percentile', args.act_percentile)
            setattr(m, 'act_range_momentum', args.act_range_momentum)
            setattr(m, 'weight_percentile', args.weight_percentile)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

            bit_conf = module_config[name]
            bitwidth = bit_conf[0] if isinstance(bit_conf, tuple) else bit_conf

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)

    # Set optimizer
    lr = args.lr if args.lr is not None else 1e-3
    optimizer = AdamW(model.parameters(), lr=lr)

    # If resume is given
    best_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
            args.start_epoch = checkpoint['epoch']
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    def validate(eval_loader, model):
        model.eval()
        losses = AverageMeter('Loss', ':.4e')
        with torch.no_grad():
            # Just one pass over eval_data
            steps = 100  # fixed number of validation steps
            for i in range(steps):
                xb, yb = eval_loader.get_batch()
                _, loss = model(xb, yb)
                losses.update(loss.item(), xb.size(0))
        return losses.avg

    def train_one_epoch(train_loader, model, optimizer, epoch, args):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            1000,
            [batch_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        model.train()
        end = time.time()
        # We'll do a fixed number of steps
        steps_per_epoch = 1000
        for i in range(steps_per_epoch):
            xb, yb = train_loader.get_batch()
            _, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), xb.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        return losses.avg

    def save_checkpoint(state, is_best, filename):
        torch.save(state, filename + 'checkpoint_gpt.pth.tar')
        if is_best:
            import shutil
            shutil.copyfile(filename + 'checkpoint_gpt.pth.tar', filename + 'model_best_gpt.pth.tar')

    # -----------------------------
    # Training Loop
    # -----------------------------
    epochs = args.epochs if args.epochs is not None else 2
    global best_acc1  # even though we don't use acc, we keep for consistency
    best_acc1 = 0

    for epoch in range(args.start_epoch, epochs):
        logging.info("Training for Epoch: {}".format(epoch))
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, args)
        logging.info("Train Loss: {:.4f}".format(train_loss))
        val_loss = validate(eval_loader, model)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path)

    logging.info("Training finished. Best val loss: {:.4f}".format(best_loss))

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained and not args.resume:
        logging.info("=> using pre-trained PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(args.arch, pretrained=True)
        if args.distill_method != 'None':
            logging.info("=> using pre-trained PyTorchCV teacher '{}'".format(args.teacher_arch))
            teacher = ptcv_get_model(args.teacher_arch, pretrained=True)
    else:
        logging.info("=> creating PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(args.arch, pretrained=False)
        if args.distill_method != 'None':
            logging.info("=> creating PyTorchCV teacher '{}'".format(args.teacher_arch))
            teacher = ptcv_get_model(args.teacher_arch, pretrained=False)

    if args.resume and not args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)['state_dict']
            model_key_list = list(model.state_dict().keys())
            for key in model_key_list:
                if 'num_batches_tracked' in key: model_key_list.remove(key)
            i = 0
            modified_dict = {}
            for key, value in checkpoint.items():
                if 'scaling_factor' in key: continue
                if 'num_batches_tracked' in key: continue
                if 'weight_integer' in key: continue
                if 'min' in key or 'max' in key: continue
                modified_key = model_key_list[i]
                modified_dict[modified_key] = value
                i += 1
            logging.info(model.load_state_dict(modified_dict, strict=False))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    quantize_arch = quantize_arch_dict[args.arch]

    for name, m in model.named_modules():
        logging.info(f"Module Old: {name}")


    model = quantize_arch(model)

    for name, m in model.named_modules():
        logging.info(f"Module New: {name}")

    bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme]
    name_counter = 0

    for name, m in model.named_modules():
        logging.info(f"Module: {name}")
        if name in bit_config.keys():
            logging.info(f"Setting quantization parameters for {name}")
            name_counter += 1
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', args.bias_bit)
            setattr(m, 'quantize_bias', (args.bias_bit != 0))
            setattr(m, 'per_channel', args.channel_wise)
            setattr(m, 'act_percentile', args.act_percentile)
            setattr(m, 'act_range_momentum', args.act_range_momentum)
            setattr(m, 'weight_percentile', args.weight_percentile)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', args.fix_BN)
            setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
            setattr(m, 'training_BN_mode', args.fix_BN)
            setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
            setattr(m, 'save_path', args.save_path)
            setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

            if type(bit_config[name]) is tuple:
                bitwidth = bit_config[name][0]
                if bit_config[name][1] == 'hook':
                    m.register_forward_hook(hook_fn_forward)
                    global hook_keys
                    hook_keys.append(name)
            else:
                bitwidth = bit_config[name]

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)

    logging.info("match all modules defined in bit_config: {}".format(len(bit_config.keys()) == name_counter))
    logging.info(model)

    if args.resume and args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading quantized checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)['state_dict']
            modified_dict = {}
            for key, value in checkpoint.items():
                if 'num_batches_tracked' in key: continue
                if 'weight_integer' in key: continue
                if 'bias_integer' in key: continue

                modified_key = key.replace("module.", "")
                modified_dict[modified_key] = value
            model.load_state_dict(modified_dict, strict=False)
        else:
            logging.info("=> no quantized checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if args.distill_method != 'None':
                teacher.cuda(args.gpu)
                teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.distill_method != 'None':
                teacher.cuda()
                teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            # teacher is not alexnet or vgg
            if args.distill_method != 'None':
                teacher = torch.nn.DataParallel(teacher).cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            if args.distill_method != 'None':
                teacher = torch.nn.DataParallel(teacher).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume optimizer and meta information from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                         format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_resolution = 224
    if args.arch == "inceptionv3":
        train_resolution = 299

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(train_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataset_length = int(len(train_dataset) * args.data_percentage)
    if args.data_percentage == 1:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    else:
        partial_train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                                 [dataset_length, len(train_dataset) - dataset_length])
        train_loader = torch.utils.data.DataLoader(
            partial_train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_resolution = (256, 224)
    if args.arch == 'inceptionv3':
        test_resolution = (342, 299)

    # evaluate on validation set
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(test_resolution[0]),
            transforms.CenterCrop(test_resolution[1]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.distill_method != 'None':
            train_kd(train_loader, model, teacher, criterion, optimizer, epoch, val_loader,
                     args, ngpus_per_node, dataset_length)
        else:
            # train(train_loader, model, criterion, optimizer, epoch, args)
            train_with_ortho_loss(train_loader, model, criterion, optimizer, epoch, args)

        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
        if is_best:
            # record the best epoch
            best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save_path)


from torch.nn.functional import normalize

def l2_reg_ortho_mat(mat, device):
    """
    SRIP function from 'Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?,' 
    https://arxiv.org/abs/1810.09102.
    """
    W = mat
    cols = W[0].numel()
    w1 = W.view(-1,cols)
    wt = torch.transpose(w1,0,1)
    m  = torch.matmul(wt,w1)
    ident = Variable(torch.eye(cols,cols)).type(torch.HalfTensor).to(device)

    w_tmp = (m - ident)
    height = w_tmp.size(0)
    u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
    v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
    u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
    sigma = torch.dot(u, torch.matmul(w_tmp, v))
    return (sigma)**2

def train_with_ortho_loss(train_loader, model, criterion, optimizer, epoch, args, lambda_ortho=0.01):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        main_loss = criterion(output, target)

        # Compute orthogonality penalty for each layer
        ortho_loss = 0
        for name, param in model.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Apply to weight matrices only
                ortho_loss += l2_reg_ortho_mat(param, device=param.device)

        # Combine main loss and orthogonality penalty
        total_loss = main_loss + lambda_ortho * ortho_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(total_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_kd(train_loader, model, teacher, criterion, optimizer, epoch, val_loader, args, ngpus_per_node,
             dataset_length):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()
    teacher.eval()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if args.distill_method != 'None':
            with torch.no_grad():
                teacher_output = teacher(images)

        if args.distill_method == 'None':
            loss = criterion(output, target)
        elif args.distill_method == 'KD_naive':
            loss = loss_kd(output, target, teacher_output, args)
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if i % args.print_freq == 0 and args.rank == 0:
            print('Epoch {epoch_} [{iters}]  Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(epoch_=epoch, iters=i,
                                                                                               top1=top1, top5=top5))

        if i % ((dataset_length // (
                args.batch_size * args.evaluate_times)) + 2) == 0 and i > 0 and args.evaluate_times > 0:
            acc1 = validate(val_loader, model, criterion, args)

            # switch to train mode
            if args.fix_BN == True:
                model.eval()
            else:
                model.train()

            # remember best acc@1 and save checkpoint
            global best_acc1
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.save_path)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
                'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
                'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
                'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
                'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
                }, args.save_path + 'quantized_checkpoint.pth.tar')

    unfreeze_model(model)

    return top1.avg


def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def loss_kd(output, target, teacher_output, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """
    alpha = args.distill_alpha
    T = args.temperature
    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(output, target) * (1. - alpha)

    return KD_loss

if __name__ == '__main__':
    main()
