import torch

from torch.utils.data import Dataset
from tree_ar import TransformerScanModel, AssociativeRecallDataset
# from data.associative_recall import multiquery_ar

from transformers import GPT2Config

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
model = TransformerScanModel(config, chunk_size, T1_num_layers=1, T2_num_layers=1) #, track_eranks=True
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
# test_data.to(device)

def predict(model, inputs):
    # Initialize states
    L = None
    chunks_processed = 0
    prefix_val = None
    past_key_values = None
    # Predict each token in the sequence
    predictions = torch.empty_like(inputs, dtype=torch.long)
    for i in range(input_seq_len):
        # Pass the sequence up to the current token
        logits, L, chunks_processed, prefix_val, past_key_values = model.forward_inference(
            inputs[:, :i+1], L, chunks_processed, prefix_val, past_key_values
        )
        # Get the predicted token (argmax of logits)
        predicted_token = torch.argmax(logits, dim=-1)
        predictions[:, i] = predicted_token

    return predictions

# Evaluate the model on the test dataset
def test_model(model, dataset):
    #inputs, labels = dataset # Assuming the dataset returns a tuple of (input_data, target)
    valid = torch.where(dataset.labels != -100)
    pred = predict(model, dataset.inputs)
    # print("Inputs: " + str(dataset.inputs))
    # print("Labels: " + str(dataset.labels))
    # print("model: " + str(model(dataset.inputs).logits.shape))

    ### predict using model()
    #pred = torch.argmax(model(dataset.inputs).logits, dim=-1)
    # print("Pred: " + str(pred))
    # print("Shape: " + str(pred.shape))

    ### predict using model.forward_inference()
    pred = predict(model, dataset.inputs)
    print("Pred: " + str(pred))
    pred = pred[valid] # do I need to offset this?
    labels = dataset.labels[valid]
    # print("Labels: " + str(labels))
    # print("Pred: " + str(pred))
    # Calculate the number of correct predictions
    correct = (pred == labels).sum() # & labels != -100
    # print("correct: " + str(correct))
    total = labels.numel() #& labels != -100
    print(str(total))
    
    # Calculate accuracy
    accuracy = correct / total
    return accuracy

# Calculate and print the accuracy
accuracy = test_model(model, test_data)
print(f"Model accuracy on test dataset: {accuracy * 100:.2f}%")