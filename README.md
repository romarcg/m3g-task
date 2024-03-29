# NEUI Team - MEDIQA-M3G: Multilingual & Multimodal Medical Answer Generation

> **neui** team
> - romarcg
> - owlmx

This repository contains the code for the NEUI team that participated in the Multilingual & Multimodal Medical Answer Generation task. 


Task description task: https://sites.google.com/view/mediqa2024/mediqa-m3g


## Solution description


### Solution Description:
We utilized a compact Visual Language Model (VLM) named Moondream to evaluate the performance of small Language-Image Models (LIMs) on the M3G multimodal task. Moondream is built upon a Sigmoid loss for Language-Image Pre-training (SigLIP) and the Phi-1.5 language model, a Transformer with 1.3 billion parameters. We fine-tuned the VLM using the provided training data, extending each case title and description to all the provided images. We employed the flash attention algorithm to mitigate memory issues during training and inference. Our hardware setup was limited to a single NVIDIA RTX 3090 GPU for fine-tuning and inference.

### Output Management:
Our solution was tested on the non-fine-tuned VLM (baseline) and the M3G fine-tuned model. We constructed the output by performing inference on each image of each case in the test dataset. This step means that for one case, we request the VLM with our query and each of the case's images. The response for this case was the concatenation of the case's responses.

### Improving Output:
The output of the previous approach might contain redundancies and short answers that deviate from the provided context in the query. To address this issue, we implemented a post-processing step of the VLM output by constructing a new query for an LLM. This step relies on the idea that we already have the context to improve the VLM answer. The context consists of the original query title and content from the test dataset cases and the VLM response, which we refer to as image analysis. Along with the context, we used the general query, "What is the disease present in the photo? What is the treatment?"

### Additional Improvements:
The previous pipeline was built considering only one language data stream, English. The VLM was fine-tuned using only the English queries, content, and responses. However, as the LLM we employed in the last step has some multilingual capabilities, we rebuilt the post-processing step of the pipeline by changing the query context to their Spanish versions. We kept the English image analysis in the context. We added additional prompt instructions to the model to request Spanish responses. As a result of this change, we could provide additional output for the Spanish language.

As the post-processing large language model (LLM), we utilized a fine-tuned [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B-DARE) version we trained during our participation in the **mediqa-corr** task. For more details check our [medicorr repository](https://github.com/OWLmx/mediqa2024_medicorr).


> Model references:
> - Moondream: https://huggingface.co/vikhyatk/moondream2
> - SigLIP model: https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384
> - Phi1.5 model: https://huggingface.co/microsoft/phi-1_5


### Submission runs:

- English: `data/submissions/run_bs_ft_vq1_en_v1.json`
- Spanish: `data/submissions/run_bs_ft_vq1_es_v2.json`


## Code requirements

Check the `requirements.txt` file.

```
conda create m3g
conda activate m3g
pip install -r requirements.txt
```

### Folder structure

- .
- - checkpoints/
- - data/
  - - M3G/
    - - mediqa-m3g/
    - - - mediqa-m3-clinicalnlp2024/
      - - - train.json
          - valid.json
      - m3g-test-allinputs-v2/
  - - images/
  - - - train/
      - test/
      - valid/
    - submissions/
- - data_processing.py
- - fine_tuning.py
- - output_generation.npyib


## Data pre-processing

This command will read the original M3G dataset files and built our own for finetuning. Check script to enable or disable image resizing in case your VRAM is not enough during the finetuning process even when chaging `BATCH_SIZE` and `GRAD_ACCUM_STEPS` parameters.

`python data_processing.py`

> We include the original json files for all datasets. Images are missing. Copy the images in their respective folder. If resize is needed, remember to named the original images folders with the prefix `BASENAME_org` and enable the resizing instructions in the data_processing.py script.

## Finetuning

This script will read the pre-processed data store in data, this incldues the images in their original or resized version.

`python fine_tuning.py`

The finetuned model will be stored in `checkpoints` folder.

## LLM post-processing step

See [medicorr repository](https://github.com/OWLmx/mediqa2024_medicorr).

## Inference & output generation

The `output_generation.ipynb` jupyter notebook shows how to process the test dataset using the different approaches in the solution description. For the LLM post-processing steps (English & Spanish), the notebook shows how to create the intermediate json file sent to our CORR LLM solution and how the `tvs` returned by it is processed to generate the submissions.

We include the original `json` submission files in the `data/submissions` folder.