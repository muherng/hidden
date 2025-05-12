import torch
import torch.nn as nn

from torch.utils.data import Dataset
from tree_ar import TransformerScanModel
from tree_ar2 import AssociativeRecallDataset

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
print("Device:", device)

# Small model for testing
chunk_size = 2
vocab_size = 100
num_examples = 2
input_seq_len = 8
num_kv_pairs = 2
seed = 117
config = GPT2Config(
        vocab_size=vocab_size, #tokenizer.vocab_size
        n_positions=1024,
        n_embd=8, #768,
        n_layer=2, #6,
        n_head=1,
        dropout=0.1
    )
class T1(nn.Module):
    """
    T1: Aggregation module.
    Uses GPT-2 blocks (without a causal mask) to aggregate two sequences.
    """
    def __init__(self, config, num_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.att = []

    def forward(self, x):
        for block in self.blocks:
            x, att = block(x, attention_mask=None, use_cache=False, output_attentions=True) #[0] , att
            # print("x: " + str(x))
            # print("att: " + str(att))
            self.att.append(att)
        x = self.ln_f(x)
        return x

class T2(nn.Module):
    """
    T2: Autoregressive prediction module.
    Uses GPT-2 blocks with a causal mask.
    """
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.att = []

    def forward(self, x, causal_mask, past_key_values=None):
        new_past = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            x, present, att = block(x, attention_mask=causal_mask, use_cache=True, output_attentions=True, layer_past=past) #order?
            self.att.append(att)
            new_past.append(present)
        x = self.ln_f(x)
        return x, tuple(new_past)

class AnalyzeTreeModel(TransformerScanModel):
    """Tree model with direct output supervision."""
    
    def __init__(self, config, chunk_size= 32, T1_num_layers = 1, T2_num_layers = 1):
        super().__init__(config, chunk_size=chunk_size, T1_num_layers = T1_num_layers, T2_num_layers = T2_num_layers, train_mode="sequential")
        self.T1 = T1(config, num_layers=T1_num_layers)
        self.T2 = T2(config, num_layers=T2_num_layers)
        self.eranks = []
    
    def erank(self, x):
        # Calculate the effective rank of the input tensor
        s = torch.linalg.svdvals(x)
        s = s / torch.sum(s)
        H = torch.distributions.Categorical(probs=s).entropy()
        return torch.exp(H)

    def combine(self, x, y):
        erank = self.erank(torch.cat([x[0], y[0]], dim=1))
        print("erank: " + str(erank))
        out, dummy = super().combine(x, y)
        self.eranks.append(torch.stack((erank, self.erank(out))))
        return out, dummy

model = AnalyzeTreeModel(config, chunk_size, T1_num_layers=1, T2_num_layers=1) #, track_eranks=True
model.eval()

# Define the configuration for the TransformerScan model
# chunk_size = 64
# vocab_size = 8192
# num_examples = 1000
# input_seq_len = 128
# num_kv_pairs = 8
# seed = 117
# config = GPT2Config(
#         vocab_size=8192, #tokenizer.vocab_size
#         n_positions=1024,
#         n_embd=128, #768,
#         n_layer=2, #6,
#         n_head=1,
#         dropout=0.1
#     )

# # Load the pre-trained TransformerScan model
# model_path = "out/mqar_128/tree_model_128/tree_20250509_110213/checkpoint-200000" #tree_20250507_174245
# model = TransformerScanModel.from_pretrained(model_path, config, chunk_size, T1_num_layers=2, T2_num_layers=2, device=device) #, track_eranks=True
# model.eval()

# Create the test dataset
test_data = AssociativeRecallDataset(   # multiquery_ar(
    vocab_size=vocab_size,
    num_examples=num_examples,
    input_seq_len=input_seq_len,
    seed=seed,
    power_a=0.01,
    num_kv_pairs=num_kv_pairs,
)

outputs = model(test_data.inputs) #.to(device)
eranks = torch.stack(model.eranks)
eranks_ratio = eranks[:, 1] / eranks[:, 0] #clean up occasions when combine does nothing?
print("Effective rank ratio: " + str(eranks_ratio.mean(-1)))

T1_attentions = model.T1.att
T2_attentions = model.T2.att
# print(f"Num layers: {len(T1_attentions)}")
# print(f"Shape of attention from layer 0: {T1_attentions[0].shape}")
print("T1 attentions: " + str(T1_attentions))
print("T2 attentions: " + str(T2_attentions))
# # test_data.to(device)

# def predict(model, inputs):
#     # Initialize states
#     L = None
#     chunks_processed = 0
#     prefix_val = None
#     past_key_values = None
#     # Predict each token in the sequence
#     predictions = torch.empty_like(inputs, dtype=torch.long)
#     for i in range(input_seq_len):
#         # Pass the sequence up to the current token
#         logits, L, chunks_processed, prefix_val, past_key_values = model.forward_inference(
#             inputs[:, :i+1], L, chunks_processed, prefix_val, past_key_values
#         )
#         # Get the predicted token (argmax of logits)
#         predicted_token = torch.argmax(logits, dim=-1)
#         predictions[:, i] = predicted_token

#     return predictions

# # Evaluate the model on the test dataset
# def test_model(model, dataset):
#     #inputs, labels = dataset # Assuming the dataset returns a tuple of (input_data, target)
#     valid = torch.where(dataset.labels != -100)
#     pred = predict(model, dataset.inputs)
#     # print("Inputs: " + str(dataset.inputs))
#     # print("Labels: " + str(dataset.labels))
#     # print("model: " + str(model(dataset.inputs).logits.shape))

#     ### predict using model()
#     #pred = torch.argmax(model(dataset.inputs).logits, dim=-1)
#     # print("Pred: " + str(pred))
#     # print("Shape: " + str(pred.shape))

#     ### predict using model.forward_inference()
#     pred = predict(model, dataset.inputs)
#     print("Pred: " + str(pred))
#     pred = pred[valid] # do I need to offset this?
#     labels = dataset.labels[valid]
#     # print("Labels: " + str(labels))
#     # print("Pred: " + str(pred))
#     # Calculate the number of correct predictions
#     correct = (pred == labels).sum() # & labels != -100
#     # print("correct: " + str(correct))
#     total = labels.numel() #& labels != -100
#     print(str(total))

#     # Calculate accuracy
#     accuracy = correct / total
#     return accuracy

# # Calculate and print the accuracy
# accuracy = test_model(model, test_data)
# print(f"Model accuracy on test dataset: {accuracy * 100:.2f}%")