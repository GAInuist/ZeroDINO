from argparse import ArgumentParser

def get_config_parser():
    parser = ArgumentParser()
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help='The GPU index')
    parser.add_argument('--DATASET', default='CUB', choices=['AWA2', 'CUB', 'SUN'], type=str, help='Dataset type')
    parser.add_argument('--DATASET_path', default='CUB/CUB_200_2011/CUB_200_2011/images/', type=str,
                        help='The path of DATASET')
    parser.add_argument('--attr_num', default=312, type=int, help='Attribute number for your Dataset')
    parser.add_argument('--is_train', default=False, type=bool, help='Whether to start training')
    parser.add_argument('--is_test', default=True, type=bool, help='Whether to start testing')
    parser.add_argument('--cs', default=False, type=bool, help='Use calibrated stacking')
    parser.add_argument('--gamma', default=0.75, type=float, help='Pre-specified gamma for calibrated stacking')
    parser.add_argument('--search_gamma', type=lambda x: str(x).lower() == 'true', default=True,
                        help='whether to search a list of gamma each epoch')
    parser.add_argument('--Num_Epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--test_batch', default=64, type=int, help='Batch size for testing')
    parser.add_argument('--pretrain_path', default=None, type=str, help='Path to pre-trained model')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of worker threads')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    parser.add_argument('--pretrained', default=True, type=bool, help='Use pretrained ViT model')
    parser.add_argument('--freeze_vit', default=False, type=bool, help='Freeze ViT parameters')
    parser.add_argument('--glove_vector_length', default=300, type=int, help='Dimension of glove vector length')
    parser.add_argument('--encoder_dim', default=768, type=int, help='Encoder dimension for Aggregation_ViT, 768, 1024')
    parser.add_argument('--num_heads', default=8, type=int, help='Number of heads in Transformer encoder')
    parser.add_argument('--ffn_hidden_dim', default=1024, type=int, help='Hidden dimension of FFN in Transformer encoder')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate in Transformer encoder')
    parser.add_argument('--fusion_layers', default=1, type=int, help='Number of Transformer encoder layers for fusion')
    parser.add_argument('--zsl_only', action='store_true',help='Just test CZSL（unseen-only），Without testing GZSL/H')
    return parser