from datasets import load_dataset

# load fineweb-edu with parallel processing
#dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64, cache_dir="/data/lingo/morrisyau/hidden/data/")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=64, cache_dir="/data/lingo/morrisyau/hidden/data/")
