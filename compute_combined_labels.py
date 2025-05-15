import argparse
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Label images using DeepSeek API + Amazon Rekognition Image.')
    parser.add_argument('--dataset_path', type=str, default='image_caption_children_in_the_wild_dataset.csv')
    parser.add_argument('--amazon_rule', type=str, default='min-range', help="Amazon Rekognition Image rule. Should be one of `min-range` or `mid-range`.")
    return parser.parse_args()


def evaluate_combined_methods(args):
    # Load the labeled dataset.
    dataset = pd.read_csv(args.dataset_path)

    # Load results of the DeepSeek method.
    with open('deepseek_labels.pickle', 'rb') as f:
        deepseek_responses = pickle.load(f)

    # Load results of the Amazon method.
    with open('amazon_labels.pickle', 'rb') as f:
        amazon_responses = pickle.load(f)

    y_true, y_pred = [], []
    for i, row in dataset.iterrows():
        if i in deepseek_responses and i in amazon_responses:
            # Extract the ground truth.
            if row['label'] == 'Final_Child':
                y_true.append(1)
            elif row['label'] == 'Final_NoChild':
                y_true.append(0)
            else:
                raise ValueError("ERROR: One of the methods labeled a sample with the Disagreement label. This should not have happened.")
            
            # Extract DeepSeek prediction.
            deepseek_response = deepseek_responses[i]
            if deepseek_response == 'yes':
                y_deepseek = 1
            elif deepseek_response == 'no':
                y_deepseek = 0
            else:
                # Invalid response.
                y_deepseek = 0

            # Extract Amazon Rekognition prediction.
            amazon_response = amazon_responses[i]
            has_child = False
            for faceDetail in amazon_response['FaceDetails']:
                if args.amazon_rule == 'min-range':
                    age_pred = faceDetail['AgeRange']['Low']
                elif args.amazon_rule == 'mid-range':
                    age_pred = (faceDetail['AgeRange']['Low'] + faceDetail['AgeRange']['High']) / 2
                else:
                    raise ValueError(f"ERROR: Unknown --amazon_rule={args.amazon_rule}. Should be one of `min-range` or `mid-range`.")
                if age_pred < 18:
                    has_child = True
                    break
            if has_child:
                y_amazon = 1
            else:
                y_amazon = 0
            y_pred.append(max(y_deepseek, y_amazon))

    assert len(y_true) == len(y_pred)
    print(f"Found {len(y_pred)} commonly labeled samples")

    cm = confusion_matrix(y_true, y_pred)
    # cm[i,j] is the number of observations known to be in group i and
    # predicted to be in group j.
    tpr = cm[1,1] / (cm[1,1] + cm[1,0])
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])
    print(f"Combined DeepSeek API with Amazon Rekognition Image using the {args.amazon_rule} rule.\nTPR={tpr:.3f}, FPR={fpr:.3f}")
 

if __name__ == '__main__':
    args = parse_args()
    evaluate_combined_methods(args)
