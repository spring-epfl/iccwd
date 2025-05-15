import argparse
import boto3
import botocore
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Label images using Amazon Rekognition Image.')
    parser.add_argument('--dataset_path', type=str, default='image_caption_children_in_the_wild_dataset.csv')
    parser.add_argument('--images_path', type=str, default='dataset')
    return parser.parse_args()


def run_age_estimation(args, region):

    # Load all the indexes of downloaded images.
    assert os.path.exists(args.images_path)
    downloaded_images = os.listdir(args.images_path)
    image_idxs = sorted([int(image_path[:-4]) for image_path in downloaded_images])
    assert len(downloaded_images) == len(image_idxs)
    image_idxs = set(image_idxs)

    # Load all the ground-truth labels.
    dataset = pd.read_csv(args.dataset_path)
    #dataset = dataset.head(1000)
    idx_to_label = dict()
    num_child, num_nochild = 0, 0
    num_disagreement, num_failed = 0, 0
    for i, row in dataset.iterrows():
        # Discard samples that were not downloaded.
        if i not in image_idxs:
            num_failed += 1
            continue
        if row['label'] == 'Disagreement':
            num_disagreement += 1
            continue
        if row['label'] == 'Final_Child':
            idx_to_label[i] = 1
            num_child += 1
        elif row['label'] == 'Final_NoChild':
            idx_to_label[i] = 0
            num_nochild += 1
        else:
            raise ValueError('ERROR: Unknown label.')
    assert num_failed == len(dataset) - len(image_idxs)
    print(f"Found {len(image_idxs)}/{len(dataset)} downloaded samples.")
    print(f"Found {len(idx_to_label)}/{len(image_idxs)} downloaded samples with Child/NoChild label. The other {num_disagreement}/{len(image_idxs)} samples have Disagreement label and will not be counted towards the metrics.")
    print(f"Found {num_child}/{len(idx_to_label)} downloaded images with Child label and {num_nochild}/{len(idx_to_label)} downloaded images with NoChild label.")
    
    session = boto3.Session(profile_name='default',
                            region_name=region)
    client = session.client('rekognition', region_name=region)

    save_path = "amazon_labels.pickle"
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            responses = pickle.load(f)
        print(f"Found {len(responses)} already labeled captions.")
    else:
        responses = dict()

    print(f"Labeling images {len(responses)} to {len(idx_to_label)}")
    # Only iterate over samples that were downloaded.
    for i, idx in tqdm(enumerate(idx_to_label), total=len(idx_to_label)):
        if i < len(responses):
            continue
        image_path = os.path.join(args.images_path, f"{idx}.jpg")
        #print(image_path)
        try:
            response = client.detect_faces(Image={'Bytes': open(image_path, 'rb').read()}, Attributes=['AGE_RANGE'])
        except botocore.exceptions.ClientError as error:
            print("Image is too large for Amazon Rekognition Image. Resizing image.")
            image = Image.open(image_path)
            width, height = image.size
            max_size = max(width, height)
            new_width = int(width * 2048/max_size)
            new_height = int(height * 2048/max_size)
            print(f"Resized the image from {width, height} to {(new_width, new_height)}")
            image = image.resize((new_width, new_height))
            image_temp_path = f"{idx}_resized_temp.jpg"
            assert not os.path.exists(image_temp_path)
            print(f"Saving resized image temporarily to {image_temp_path}")
            image.save(image_temp_path)
            response = client.detect_faces(Image={'Bytes': open(image_temp_path, 'rb').read()}, Attributes=['AGE_RANGE'])
            os.remove(image_temp_path)
            print(f"Removing the temporarily saved resized image from {image_temp_path}")

        assert idx not in responses
        responses[idx] = response

        if i % 100 == 0 and i > 0:
            with open(save_path, 'wb') as f:
                pickle.dump(responses, f)

    with open(save_path, 'wb') as f:
        pickle.dump(responses, f)

    return idx_to_label, responses


def evaluate_responses(args, idx_to_label, responses): 
    assert len(idx_to_label) == len(responses)
    y_true, y_pred, age_preds = defaultdict(list), defaultdict(list), \
            defaultdict(list)

    # Prepare the ROC curve figure.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = ['orange', 'blue']
    titles = ['Min-range rule', 'Mid-range rule']
    
    for j, method in enumerate(['min-range', 'mid-range']):
        for idx, label in idx_to_label.items():
            assert idx in responses
            y_true[method].append(label)

            response = responses[idx]
            has_child = False
            min_age = 1000
            for faceDetail in response['FaceDetails']:
                if method == 'min-range':
                    age_pred = faceDetail['AgeRange']['Low']
                elif method == 'mid-range':
                    age_pred = (faceDetail['AgeRange']['Low'] + faceDetail['AgeRange']['High']) / 2
                else:
                    raise ValueError("ERROR: Unknown method {args.method}.")
                if age_pred < 18:
                    has_child = True
                min_age = min(min_age, age_pred)
            if has_child:
                y_pred[method].append(1)
            else:
                y_pred[method].append(0)
            age_preds[method].append(-min_age)
        assert len(y_true[method]) == len(y_pred[method])

        cm = confusion_matrix(y_true[method], y_pred[method])
        # cm[i,j] is the number of observations known to be in group i and
        # predicted to be in group j.
        tpr = cm[1,1] / (cm[1,1] + cm[1,0])
        fpr = cm[0,1] / (cm[0,0] + cm[0,1])
        print(f"Method={method}, TPR={tpr:.3f}, FPR={fpr:.3f}")

        all_fpr, all_tpr, all_thresholds = roc_curve(y_true[method], 
                age_preds[method])
        t = 0
        for i in range(len(all_fpr)):
            ti = - all_thresholds[i]
            if ti < 18:
                t = ti
            else:
                break

        ax.plot(all_fpr[:-1], all_tpr[:-1], label=titles[j], color=colors[j])
        ax.plot(all_fpr[i-1], all_tpr[i-1], marker='o', color=colors[j])
        ax.hlines(y=all_tpr[i-1], xmin=1e-4, xmax=all_fpr[i-1], color=colors[j], ls=':', lw=1)
        ax.vlines(x=all_fpr[i-1], ymin=0, ymax=all_tpr[i-1], color=colors[j], ls=':', lw=1)
        if j==1:
            ax.text(0.01, 0.7, 'age=18', color='black', fontsize=12)
    
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylim(1e-4, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig('amazon_roc_curve.pdf')
    plt.close()
    
def main():
    region='us-east-1'
    args = parse_args()
    idx_to_label, responses = run_age_estimation(args, region)
    evaluate_responses(args, idx_to_label, responses)

if __name__ == "__main__":
    main()

