import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PAD = 28 


class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.src_mask = None
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size,padding_idx=PAD)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4, 
        num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask= True):
        # src: (T,B)
    
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):      
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
            
        embedded = self.embed(src)  # (T,B,H)
        embedded += self.pe(embedded) # (T,B,H)
        hidden = self.trfm(embedded, embedded, src_mask=self.src_mask) # (T,B,H)
        out = self.out(hidden) # (T,B,V)
        out = F.log_softmax(out, dim=2) # (T,B,V)
        return out # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded += self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)

        return output
    
    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size<=100:
            return self._encode(src)
        else: # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st,ed = 0,100
            out = self._encode(src[:,st:ed]) # (B,4H)
            while ed<batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:,st:ed])], axis=0)
            return out

    def _decode(self, src):
        output = src
        for i in range(self.trfm.decoder.num_layers - 1):
            output = self.trfm.decoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().numpy()
        output = self.trfm.decoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.decoder.norm:
            output = self.trfm.decoder.norm(output) # (T,B,H)
        output = output.detach().numpy()

        return output
        
    def decode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size<=100:
            return self._decode(src)
        else: # Batch is too large to load
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st,ed = 0,100
            out = self._decode(src[:,st:ed]) 
            while ed<batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._decode(src[:,st:ed])], axis=0)
            return out       
        
        
def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=15, help='number of epochs')
    parser.add_argument('--vocab', '-v', type=str, default='vocab_train.txt', help='vocabulary (.pkl)')
    parser.add_argument('--data', '-d', type=str, default='moses_zinc_train250K.npy', help='train corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='ST', help='model name')
    parser.add_argument('--seq_len', type=int, default=57, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


def evaluate(model, test_loader, vocab_length):
    model.eval()
    total_loss = 0
    for b, sm in enumerate(test_loader):
        sm = torch.t(sm.cuda()) # (T,B)
        with torch.no_grad():
            output = model(sm) # (T,B,V)
        loss = F.nll_loss(output.view(-1, vocab_length), 
                               sm.contiguous().view(-1),
                               ignore_index=PAD)
        total_loss += loss.item()
    return total_loss / len(test_loader)


def get_train_dataset(file,sequence_length=57):

    train_samples = np.load(file, allow_pickle=True)  

    sample_size = train_samples.size

    processed_sample=[]

    for index in range(0, sample_size):
        sample = train_samples[index]
        if len(sample) < sequence_length:
            sample = np.concatenate((sample, np.full(sequence_length-len(sample), PAD)))
        else:
            sample = np.resize(sample, sequence_length)
     
        processed_sample.append(sample) 
    return processed_sample

def get_vocab(file):

    vocab = pd.read_csv(file, delimiter=" ", header=None).to_dict()[0]
    vocab = dict([(v,k) for k,v in vocab.items()])
    
    return vocab

def main():
    args = parse_arguments()
    assert torch.cuda.is_available()
    
    print('Loading dataset...')

    vocab = get_vocab(args.vocab)
    vocab_length = len(vocab)
    
    seq_length = args.seq_len
    
    dataset = get_train_dataset(args.data,seq_length)
    
    test_size = 20000
    train, test = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    print('Train size:', len(train))
    print('Test size:', len(test))
    
    del dataset, train, test

    model = TrfmSeq2seq(vocab_length, args.hidden, vocab_length, args.n_layer).cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #print(model)
    print('Total parameters:', sum(p.numel() for p in model.parameters()))

    best_loss = None
    for e in range(1, args.n_epoch):
        for b, sm in tqdm(enumerate(train_loader)):
            sm = torch.t(sm.cuda()) # (T,B)
            optimizer.zero_grad()
            output = model(sm) # (T,B,V)
            loss = F.nll_loss(output.view(-1, vocab_length), 
                    sm.contiguous().view(-1), ignore_index=PAD)
            loss.backward()
            optimizer.step()
            if b%500==0:
                print('Train {:3d}: iter {:5d} | loss {:.7f} | ppl {:.7f}'.format(e, b, loss.item(), math.exp(loss.item())))

            if b%10000==0:
                loss = evaluate(model, test_loader, vocab_length)
                print('Val {:3d}: iter {:5d} | loss {:.7f} | ppl {:.7f}'.format(e, b, loss, math.exp(loss)))

                # Save the model if the validation loss is the best we've seen so far.
                if not best_loss or loss < best_loss:
                    print("[!] saving model...")

                    if not os.path.isdir(".save"):
                        os.makedirs(".save")
                    torch.save(model.state_dict(), './.save/trfm_new_%d_%d.pkl' % (e,b))
                    best_loss = loss

   
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

