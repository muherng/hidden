import torch
import torch.nn as nn

from torch.utils.data import Dataset
from tree import TransformerScanModel
from tree_ar2 import AssociativeRecallDataset
from tree_copy import CopyDataset

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
print("Device:", device)

def get_accuracy(outputs, labels): 
    logits = outputs["logits"]
    print(outputs.keys())
    # labels = outputs["labels"]
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    # labels = torch.tensor(labels)

    # Flatten if needed
    if preds.ndim > 1:
        preds = preds.view(-1)
        labels = labels.view(-1)

    mask = labels != -100  # Ignore padding or masked labels
    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

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
        super().__init__(config, chunk_size=chunk_size, T1_num_layers = T1_num_layers, T2_num_layers = T2_num_layers) # , train_mode="sequential"
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
        # print("erank: " + str(erank))
        out, dummy = super().combine(x, y)
        self.eranks.append(torch.stack((erank, self.erank(out))))
        return out, dummy

def entropy(x):
    # element wise entropy
    return -x * torch.log(x)

# # Small model for testing
# chunk_size = 2
# vocab_size = 100
# num_examples = 2
# input_seq_len = 8
# num_kv_pairs = 2
# seed = 117
# config = GPT2Config(
#         vocab_size=vocab_size, #tokenizer.vocab_size
#         n_positions=1024,
#         n_embd=8, #768,
#         n_layer=2, #6,
#         n_head=1,
#         dropout=0.1
#     )
    
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

# config = GPT2Config(
#         vocab_size=8192, #tokenizer.vocab_size
#         n_positions=1024,
#         n_embd=384, #128,
#         n_layer=4, #2, #6,
#         n_head=6, #1, #12,
#         dropout=0.1
#     )
# input_seq_len = 256 # 128 #

chunk_size = 32 #256 # 16 # input_seq_len
num_examples= 5
num_kv_pairs = 4

# Load the pre-trained TransformerScan model
# model_path = "out/copy_128/tree_model_128/tree_20250513_161249/checkpoint-3125"
# model_path = "out/copy_128/tree_model_128/trafo/tree_20250513_160344/checkpoint-3125" #tree_20250507_174245
model_path = "out/mqar_128/tree_model_384/tree_20250514_163721/tree_20250514_163721/checkpoint-200000"
# model_path = "out/mqar_128/tree_model_384/trafo_20250514_223812/tree_20250514_223812/checkpoint-200000" #tree_20250507_174245
# model = TransformerScanModel.from_pretrained(model_path, config, chunk_size, T1_num_layers=config.n_layer, T2_num_layers=config.n_layer, device=device) #, track_eranks=True


# # Create the test dataset
# test_data = AssociativeRecallDataset(   
#     vocab_size=8192,
#     num_examples=num_examples,
#     input_seq_len=input_seq_len,
#     seed=56,
#     power_a=0.01,
#     num_kv_pairs=num_kv_pairs,
# )

# Create the test dataset
test_data = CopyDataset(   
    num_samples=num_examples,
    seq_len=input_seq_len
)

model = AnalyzeTreeModel.from_pretrained(model_path, config, chunk_size, T1_num_layers=config.n_layer, T2_num_layers=config.n_layer) #, track_eranks=True
model.eval()
outputs = model(test_data.inputs) #.to(device)

# Calculate and print the accuracy
accuracy = get_accuracy(outputs, test_data.labels)
print(f"Model accuracy on test dataset: {accuracy * 100:.2f}%")

# ### for T1 or when Trafo:
# T1_attentions = torch.stack(model.T1.att) # shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
# H_per_token_T1 = entropy(model.T2.att).nansum(-2)
# mean_H_T1 = H_per_token.mean()
# print("Mean entropy per token T1: " + str(mean_H_T1))

### for T2 when Tree:
H_per_token_T2 = []
for att in model.T2.att:
    H_per_token_T2.append(entropy(att).nansum(-2).flatten()) # shape of each entry before flattening: (batch_size, num_heads, seq_len)
    mean_H_T2 = torch.cat(H_per_token_T2).mean()
print("Mean entropy per token T2: " + str(mean_H_T2))


# eranks = torch.stack(model.eranks)
# eranks_ratio = eranks[:, 1] / eranks[:, 0] #clean up occasions when combine does nothing?
# print("Effective rank ratio: " + str(eranks_ratio.mean(-1)))


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
