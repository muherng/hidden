Command for training an LSTM on wikitext-2 

```bash 
python train.py --cuda --epochs 20           # Train a LSTM on Wikitext-2 with CUDA.
```

The model uses the `nn.RNN` module 

During training, if a keyboard interrupt (Ctrl-C) is received, training is stopped and the current model is evaluated against the test dataset.

The `train.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --mps                 enable GPU on macOS
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the transformer model
  --dry-run             verify the code and the model
```


