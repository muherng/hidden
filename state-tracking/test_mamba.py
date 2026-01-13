import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def test_mamba():
    print("Testing Mamba installation and model loading...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "state-spaces/mamba-130m"
    
    print(f"\nLoading model from {model_name}...")
    # First load the model to get its configuration
    model = MambaLMHeadModel.from_pretrained(
        model_name,
        device=device,
        dtype=torch.float16,
    )
    
    print(f"\nLoading tokenizer...")
    # Create a minimal tokenizer with the correct vocabulary size
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    
    # Create a minimal vocabulary dictionary
    minimal_vocab = {
        "[PAD]": 0,
        "[EOS]": 1,
        "[UNK]": 2,
    }
    
    # Build a fast tokenizer
    tk_model = models.WordLevel(vocab=minimal_vocab, unk_token="[UNK]")
    tk = Tokenizer(tk_model)
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.decoder = decoders.WordPiece()
    
    # Create the Hugging Face Fast tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]"
    )
    
    # Create a simple test input
    test_text = "Hello, I am a language model"
    print(f"\nTest input: {test_text}")
    
    # Tokenize input and only keep input_ids
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    # Print some basic information
    print("\nModel information:")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Model vocabulary size: {model.config.vocab_size}")
    
    # Generate some text
    print("\nGenerating text...")
    generated = model.generate(
        input_ids=input_ids,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1
    )
    
    # Decode and print generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated text: {generated_text}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_mamba() 