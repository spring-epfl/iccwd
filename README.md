# Code for the Image-Caption in the Wild Dataset (ICCWD)

We provide the code to download images of ICCWD and evaluate three different minor detection methods on it.

The code can be run on a commodity laptop and does not require specific compute, as it queries external APIs, although 8 or more cores are preferrable for faster results from DeepSeek API.

## Dataset download

The dataset is available on HuggingFace as this [link](https://huggingface.co/datasets/amcretu/iccwd).

To download the images, create an environment using the following commands:

```
conda create -n iccwd python=3.11.8
conda activate iccwd
conda install pip

pip install datasets opencv-python pandas pdqhash pillow requests tqdm
```

Then run the following scripts:

```
python dataset_download_csv.py
python dataset_download_images_verify.py --folder=dataset
```

## Code for minor detection methods

### DeepSeek

To label the captions in the dataset using DeepSeek, install the following dependencies.

```
pip install openai scikit-learn
```

Create an account on the DeepSeek platform and top it up with a small amount, e.g., 1$ (we used less than 0.25$ to label all the captions, but the price [varies](https://api-docs.deepseek.com/quick_start/pricing) during the day). Create an [api key](https://platform.deepseek.com/api_keys), download it and save it to a text file under `api_key.txt`.

Run the following script to label the captions:

```
python compute_deepseek_labels.py --num_procs=1 --batch_size=1
```

You can adjust `--num_procs` depending on the number of cores available on your machine. Increasing it will allow the script to run faster by parallelizing the requests. Alternatively, you can replace multiprocessing with multithreading.

The script outputs the TPR and FPR.

### Amazon Rekognition Image

To label the images in the dataset using Amazon Rekognition Image, install the following dependencies.

```
pip install boto3 matplotlib
```

Then setup an AWS account as described [here](https://docs.aws.amazon.com/rekognition/latest/dg/faces-detect-images.html). The total cost for labeling 10,000 images is estimated at 12$.

Run the following script to label the images:

```
python compute_amazon_labels.py
```

The script outputs the TPR and FPR of minor detection using Amazon Rekognition Image and two different classification rules, and saves the ROC curve to `amazon_roc_curve.pdf`.

### Combined methods

To obtain results using the combined DeepSeek API with Amazon Rekognition Image, run the following script:

```
python compute_combined_labels.py --amazon_rule=min-range
```

The script outputs the TPR and FPR of the combined method using the `min-range` classification rule. You can set `--amazon_rule=mid-range` to output results using the `mid-range` classification rule.


