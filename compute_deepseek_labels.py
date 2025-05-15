import argparse
import multiprocessing as mp
import numpy as np
from openai import OpenAI
import os
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Label captions.')
    
    parser.add_argument('--api_key_path', type=str, default='api_key.txt')
    parser.add_argument('--dataset_path', type=str, default='image_caption_children_in_the_wild_dataset.csv')
    parser.add_argument('--num_procs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--split', type=str, default=None)
    return parser.parse_args()


def read_captions(args):
    # Read the captions to be labeled.
    dataset = pd.read_csv(args.dataset_path)
    #dataset = dataset.head(100)
    captions = []
    y_true = []
    for i, row in dataset.iterrows():
        if row['label'] == 'Disagreement':
            continue
        captions.append( (i, row['caption']) )
        if row['label'] == 'Final_Child':
            y_true.append(1)
        elif row['label'] == 'Final_NoChild':
            y_true.append(0)
        else:
            raise ValueError('ERROR: Unknown label.')
    #print(captions)
    print(f"Found {len(captions)} captions with child/no child label, discarding {len(dataset)-len(captions)} captions with Disagreement label.")
    print(f"Found {np.sum(y_true)} images with child label and {len(y_true)-np.sum(y_true)} images with no child label.")
    return captions, y_true


def label_captions(args):
    # Read the captions to be labeled.
    captions, y_true = read_captions(args)

    # Read the API key.
    with open(args.api_key_path, 'r') as f:
        api_key = f.read().rstrip()

    #client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = "You are a helpful assistant. Does this caption refer to a child? You must answer with yes or no." 
    save_path = os.path.join(f'deepseek_labels.pickle')
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            responses = pickle.load(f)
        print(f"Found {len(responses)} already labeled captions.")
    else:
        responses = dict()
    #print(responses)

    print(f"Labeling captions {len(responses)} to {len(captions)}")
    for pool_start in tqdm(range(len(responses), len(captions), \
            args.num_procs*args.batch_size)):
        batch_start = pool_start
        batch_end = pool_start + min(len(captions), batch_start + \
                args.num_procs*args.batch_size)
        batch_args = []
        for i in range(args.num_procs):
            batch_captions = captions[(batch_start+i*args.batch_size):\
                    (batch_start+(i+1)*args.batch_size)]
            if len(batch_captions) == 0:
                break
            batch_args.append((api_key, prompt, batch_captions))
        with mp.Pool(len(batch_args)) as pool:
            batch_responses = pool.map(label_captions_batch, batch_args)
            responses = update_responses(responses, batch_responses)
            save_responses(responses, save_path) 
    print(f"Labeled {len(responses)} captions.")
    return responses, y_true


def update_responses(responses, batch_responses):
    for batch_response in batch_responses:
        for key, response in batch_response.items():
            assert key not in responses
            responses[key] = response
    return responses


def save_responses(responses, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(responses, f)


def label_captions_batch(args):
    api_key, prompt, captions = args
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    responses = dict()
    for key, caption in captions:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Caption: " + caption},
                ],
            max_tokens=4,
            stream=False
            )
        # Extract the text output.
        responses[key] = response.choices[0].message.content
    return responses


def evaluate_responses(responses, y_true):
    # Load DeepSeek's predictions.
    with open('deepseek_labels.pickle', 'rb') as f:
        responses = pickle.load(f)

    assert len(responses) == len(y_true), 'ERROR: The dataset has not been fully labeled.'

    y_pred = []
    for i, response in responses.items():
        if response == 'yes':
            y_pred.append(1)
        elif response == 'no':
            y_pred.append(0)
        else:
            # Assign the no label (note: when we ran the method for the paper,
            # no invalid response was returned).
            y_pred.append(0)

    cm = confusion_matrix(y_true, y_pred)
    # cm[i,j] is the number of observations known to be in group i and 
    # predicted to be in group j.
    tpr = cm[1,1] / (cm[1,1] + cm[1,0])
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"TPR={tpr:.3f}, FPR={fpr:.3f}")


if __name__ == '__main__':
    args = parse_args()
    responses, y_true = label_captions(args)
    evaluate_responses(responses, y_true)

