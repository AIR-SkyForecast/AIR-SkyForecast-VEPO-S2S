from vepo_gru_preprocess.vg_feature_engineering import gen_dataloader
from argparse import Namespace
from  vepo_gru_util import train, test1, path


if __name__ == '__main__':
    args = Namespace(input_size=5, Feature_input_size=5, hidden_size=64, output_size=2, num_layers=2, decoder_num_layers=2,
                     batch_size=128, embedding_dim=8, lr=0.001,weight_decay=0.0, step_size=100, gamma=0.1, flag='s2s')
    print('hidden_size:',args.hidden_size, ' num_layers:',args.num_layers, ' batch_size:',args.batch_size,
          ' embedding_dim:',args.embedding_dim, ' lr:',args.lr)
    Dtr, Val, Dte, m, n = gen_dataloader()
    train(args, Dtr, Val, Dte)
    test1(args, Dte)



