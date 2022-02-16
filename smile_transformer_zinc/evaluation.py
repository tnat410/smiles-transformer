import argparse
import torch
from pretrain_trfm_zinc import TrfmSeq2seq
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

pad_index = 28

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--data', '-d', type=str, default='moses_zinc_train250K.npy', help='test corpus (.csv)')
    parser.add_argument('--seq_len', type=int, default=57, help='maximum length of the paired seqence')
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--vocab_size', type=int, default=30, help='size of vocabulary') 
    parser.add_argument('--model', '-m', type=str, default='trfm_new_9_0.pkl', help='trained model (.pkl)')
    parser.add_argument('--n_layer', '-l', type=int, default=4, help='number of layers')
    
    return parser.parse_args()

def get_train_dataset(file,sequence_length=57):

    train_samples = np.load(file, allow_pickle=True) 

    processed_sample=[]

    for index in range(0, train_samples.size):
        sample = train_samples[index]
        if len(sample) < sequence_length:
            sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
        else:
            sample = np.resize(sample, sequence_length)
     
        processed_sample.append(sample) 
    return processed_sample

def main():

    args = parse_arguments()
    
    vocab_length = args.vocab_size
    seq_length = args.seq_len
    hidden_size = args.hidden
    layer = args.n_layer

    processed_sample = get_train_dataset(args.data,seq_length)
    test_size = 100
    train, test = torch.utils.data.random_split(processed_sample, [len(processed_sample)-test_size, test_size])
    print('Train size:', len(train))
    print('Test size:', len(test))
 
    input_sm = torch.tensor(test).detach().numpy()

    input_seq = pd.DataFrame(input_sm)
    input_seq.to_csv("input_seq.csv",index=False,header=False)

    trfm = TrfmSeq2seq(vocab_length, hidden_size, vocab_length, layer)
    
    model_file = args.model
    trfm.load_state_dict(torch.load(model_file))
    trfm.eval()

    test = torch.tensor(test)
    out_put = trfm(torch.t(test))
    out_put = out_put.detach().numpy()
    out_put = np.transpose(out_put, (1, 0, 2))
    out_put = torch.from_numpy(out_put)

    output_sm = []
    for i in range (0,test_size):
        string = torch.argmax(out_put[i], dim=1)
        string = string.detach().numpy()
        output_sm.append(string)
        
    
    pred_seq = pd.DataFrame(output_sm)
    pred_seq.to_csv("pred_seq.csv",index=False,header=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)



