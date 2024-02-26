"""Convert ALBERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from models.transformers.modeling_albert import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert
# from model.modeling_albert_bright import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert
import logging
logging.basicConfig(level=logging.INFO)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = AlbertConfig.from_pretrained(bert_config_file)
    # print("Building PyTorch model from configuration: {}".format(str(config)))
    model = AlbertForPreTraining(config)
    # Load weights from tf checkpoint
    load_tf_weights_in_albert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,args.bert_config_file,
                                     args.pytorch_dump_path)

'''
python convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/albert_large_zh \
    --bert_config_file=./prev_trained_model/albert_large_zh/config.json \
    --pytorch_dump_path=./prev_trained_model/albert_large_zh/pytorch_model.bin
    

from model.modeling_albert_bright import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert
python convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/albert_base_bright \
    --bert_config_file=./prev_trained_model/albert_base_bright/config.json \
    --pytorch_dump_path=./prev_trained_model/albert_base_bright/pytorch_model.bin
'''