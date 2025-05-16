from datasets import load_dataset


dataset = load_dataset("amcretu/iccwd")
dataset['train'].to_csv("image_caption_children_in_the_wild_dataset.csv")
