

#CUDA_VISIBLE_DEVICES=2 python blocking.py --hidden 256 --layers 12 --heads 8 --text_run 9 --state_run 0 --seq_len 512 --mode two_stage --epochs 20

#CUDA_VISIBLE_DEVICES=2 python blocking.py --hidden 256 --layers 6 --heads 8 --text_run 8 --state_run 1 --seq_len 512 --mode two_stage --epochs 20

#!/usr/bin/env python3
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    log_dir = './logs'
    # Look for TensorBoard event files in the logs folder.
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print("No TensorBoard event files found in './logs'")
        return

    plt.figure(figsize=(10, 6))
    
    # Process each event file individually.
    labels = ['two_stage', 'one_stage']
    line = 0
    for event_file in event_files:
        print(f"Processing event file: {event_file}")
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        scalar_tags = event_acc.Tags().get('scalars', [])
        print(f"Scalar tags: {scalar_tags}")

        # Check for the eval perplexity metric using either naming convention.
        metric_name = None
        if 'eval_ppl' in scalar_tags:
            metric_name = 'eval_ppl'
        elif 'eval/ppl' in scalar_tags:
            metric_name = 'eval/ppl'
        else:
            print(f"No evaluation perplexity metric found in file: {event_file}")
            continue

        # Extract step and metric value data.
        eval_events = event_acc.Scalars(metric_name)
        steps = [event.step for event in eval_events]
        ppl_values = [event.value for event in eval_events]

        # Use the basename of the file as a label for the plot.
        #label = os.path.basename(event_file)
        plt.plot(steps, ppl_values, marker='o', label=labels[line])
        line += 1

    # Configure and show the plot.
    plt.xlabel("Global Step")
    plt.ylabel("Evaluation Perplexity")
    plt.title("Evaluation Perplexity Over Time: text/state 8:1, 256/12/8 hidden/layers/heads, wikitext-2, 20 epochs")
    plt.ylim(0, 1000)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/eval_perplexity_all_{timestamp}.png")
    print("Plot saved as eval_perplexity_all.png")
    #plt.show()
        

if __name__ == '__main__':
    main()