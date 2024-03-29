import pandas as pd
import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import uuid

import os, sys
from PIL import Image
import glob


data_path = "data/M3G"
data_path_test = "data/m3g-test-allinputs-v2"

def get_stats(json_obj):
    _stats = []
    for el in json_obj:
        mrow = [el["encounter_id"], el["author_id"], len(el["image_ids"]), 
                1 if "query_title_zh" in el else 0,
                1 if "query_content_zh" in el else 0,
                1 if "query_title_en" in el else 0,
                1 if "query_content_en" in el else 0,
                1 if "query_title_es" in el else 0,
                1 if "query_content_es" in el else 0,
                len(el["responses"])
               ]
        _stats.append(mrow)
    df_stats = pd.DataFrame(data=_stats, columns=["encounter_id", "author_id", "images", 
                                                   "query_title_zh", "query_content_zh",
                                                   "query_title_en", "query_content_en",
                                                   "query_title_es", "query_content_es",
                                                   "responses"
                                                  ])
    return df_stats


def generate_sample(jpg_image_path="image_path.jpg", human="what is shown in the image?", gpt="An image of a an skin lesion.", fixid=None):
    sample = { 
        "id": str(uuid.uuid4()) if fixid is None else fixid,
        "image": jpg_image_path,
        "conversations": [
            {
                "from": "human",
                "value": human #.replace("\n", " "),
            },
            {
                "from": "gpt",
                "value": gpt #.replace("\n", " "),
            }
        ]
    }
    return sample


def resize_dataset(base_folder):
    # NOTE: If your HW manages to load such big pictures, this step is not necessary
    # However, the original size is quite big and the effect (at least in inference) 
    # is not impactful
    
    images_path_src = f"{data_path}/images_{base_folder}_org"#f"{data_path}/images_test_org"
    images_path_dst = f"{data_path}/images_{base_folder}" #f"{data_path}/images_test"
    resize_factor = 2
    
    images_ls = glob.glob(f'{images_path_src}/*')
    print(f"Files in folder: {len(images_ls)}")
    
    r_size = (500, 500)
    
    for infile in images_ls:
        try:
            im = Image.open(infile)
            #print (im.size)
            im.thumbnail(r_size, Image.Resampling.LANCZOS)
            fname = os.path.basename(infile)
            im.save(f"{images_path_dst}/{fname}")
            #print (f"{images_path_dst}/{fname} :: {im.size}")
        except IOError:
            print ("cannot create resized image for '%s'" % infile)

def create_llava_ds(images_path, json_ds, lang="es", additional="What is the disease present in the image and the possible treatment?"):
    big_json_list = []
    for el in json_ds:
        for imgp in el["image_ids"]:
            _qtitle = el[f"query_title_{lang}"]
            _qcontent = el[f"query_content_{lang}"]
            # if no direct question, assume is asking for disease AND treatment
            _qadditional = additional

            if "responses" in el:
                for rsp in el["responses"]:
                    _response =  rsp[f"content_{lang}"]
                    sample = generate_sample(f"{images_path}/{imgp}",
                                             #human=f"<image>\n{_qtitle} {_qcontent} {_qadditional}",
                                             human=f"{_qtitle} {_qcontent} {_qadditional}",
                                             gpt=f"{_response}")
                    big_json_list.append(sample)
            else:
                _response = ""
                sample = generate_sample(f"{images_path}/{imgp}",
                                         #human=f"<image>\n{_qtitle} {_qcontent} {_qadditional}",
                                         human=f"{_qtitle} {_qcontent} {_qadditional}",
                                         gpt=f"{_response}",
                                         fixid=el["encounter_id"])
                big_json_list.append(sample)
                
    return big_json_list



def generate_test_sample(jpg_image_path=["image_path.jpg"], title="what is shown in the image?", content="Some description", fixid=None):
    sample = { 
        "id": str(uuid.uuid4()) if fixid is None else fixid,
        "image": jpg_image_path,
        "conversations": [
            {
                "from": "human",
                "value": title 
            },
            {
                "from": "human",
                "value": content 
            },
        ]
    }
    return sample
    
def create_llava_ds_test(images_path, json_ds, lang="es"):
    big_json_list = []
    for el in json_ds:
        _qtitle = el[f"query_title_{lang}"]
        _qcontent = el[f"query_content_{lang}"]
        # if no direct question, assume is asking for disease AND treatment
        _images = [f"{images_path}/{im}" for im in el["image_ids"]]

        sample = generate_test_sample(_images,
                                 #human=f"<image>\n{_qtitle} {_qcontent} {_qadditional}",
                                 title=f"{_qtitle}",
                                 content=f"{_qcontent}",
                                 fixid=el["encounter_id"])
        big_json_list.append(sample)
                
    return big_json_list


def processing():
    # json files
    with open(f"{data_path}/mediqa-m3-clinicalnlp2024/train.json","r") as f:
        json_train = json.load(f)
    
    with open(f"{data_path}/mediqa-m3-clinicalnlp2024/valid_ht.json","r") as f:
        json_valid_ht = json.load(f)

    print(f"Sample from training:")
    print(json_train[0])

    df_train = get_stats(json_train)    
    df_valid_ht = get_stats(json_valid_ht)
    
    print (df_valid_ht)

    # plot stats for datasets:
    # moved to debug notebooks
    
    # generate our datafiles
    # NOTE: file naming convention refers to the LLaVA dataset format not to the model used
    #       this format will allow us to test it on other pretrained models for validation

    print(f"Processing training and validation dataset")
    
    lang = "en"
    faux_train_image_path = "data/images/train"
    faux_valid_image_path = "data/images/valid"
    
    llava_ds_train_es = create_llava_ds(faux_train_image_path, 
                                  json_train, 
                                  lang, 
                                  #"¿Cuál es la enfermedad presente en la imagen y cuál es el posible tratamiento?")
                                  "What is the disease in the photo? What is the best treatment?")
    llava_ds_val_es = create_llava_ds(faux_valid_image_path, 
                                  json_valid_ht, 
                                  lang, 
                                  #"¿Cuál es la enfermedad presente en la imagen y cuál es el posible tratamiento?")
                                  "What is the skin disease shown in the photo? What is the best treatment?")

    with open(f"{data_path}/llava_train_{lang}_ds_ls.json", "w") as f:
        json.dump(llava_ds_train_es, f, indent=4)
    with open(f"{data_path}/llava_val_{lang}_ds_ls.json", "w") as f:
        json.dump(llava_ds_val_es, f, indent=4)

    print(f"Sample:")
    print(llava_ds_val_es[0])


    # resize images for loading convenience (resolution to this max new resolution have little impact --tested--)
    # only in case your HW cannot handle loading original size pictures for finetuning
    #resize_dataset(base_folder="train") # original images should be in data in a folder called BASE_org
    #resize_dataset("valid")
    
    
    
    # for the test dataset (some tweaks)

    print(f"Processing test dataset")

    # json files
    with open(f"{data_path_test}/input.json","r") as f:
        json_test = json.load(f)

    print(f"Test sample")
    print(json_test[0])

    # Generate the files for both languages
    
    lang = "en"
    faux_test_image_path = "data/images/test"
    
    llava_ds_test = create_llava_ds_test(faux_test_image_path, 
                                  json_test, 
                                  lang)
    with open(f"{data_path}/llava_test_{lang}_ds_ls.json", "w", encoding='utf-8') as f:
        json.dump(llava_ds_test, f, indent=4, ensure_ascii=False)

    lang = "es"
    faux_test_image_path = "data/images/test"
    
    llava_ds_test = create_llava_ds_test(faux_test_image_path, 
                                  json_test, 
                                  lang)
    with open(f"{data_path}/llava_test_{lang}_ds_ls.json", "w", encoding='utf-8') as f:
        json.dump(llava_ds_test, f, indent=4, ensure_ascii=False)

    # resize if needed (and if you used a resize version of the training images for finetuning)
    #resize_dataset("test")
    


if __name__ == "__main__":
    processing()
