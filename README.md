# DeepFake_analysis
## Paper:
**Abstract**: Owing to the swift advancement of face swapping technology utilized in creating ”Deepfakes” and the increasing public awareness regarding multimedia security, numerous studies have been undertaken in recent years to devise various methods for detecting deepfakes. This paper presents a framework aimed at enhancing clarity in deepfake detection techniques through the reconstruction of source images from a provided deepfake image. Our focus is on developing a system that not only addresses the inquiry, ”Is this image a DeepFake or not?” but also provides insights into the question, ”If the image is a DeepFake, what are the source images?” An extensive experiment was conducted on our hypothesis, yielding significant results from the exploration.
<br/><br/>
**Keywords:** DeepFake, Faceswap, DeepFake image analysis, DeepFake source images.
<br/><br/>
[Preprint]()<br/>

## Dataset:
[**DeepFake(Faceswapped Images)**](https://www.kaggle.com/datasets/syedajannatulnaim/deepfakeface-swapped-images-using-ffhq-dataset)<br/>
[**Background removed images**](https://www.kaggle.com/datasets/syedajannatulnaim/background-removed-images-of-ffhq-dataset?select=flickr_remb)<br/>
[**Segmented Face images**](https://www.kaggle.com/datasets/syedajannatulnaim/background-removed-images-of-ffhq-dataset?select=segmented_face)<br/>

## Instructions:
### To inference:
```
python3 inference.py image-path
```

### For Gradio:

app.py


### To Train and Validation:

train.py

### To test individual model:
test.py

## Huggingface Space:
[Jannat24/uncovering_deepfake_image](https://huggingface.co/spaces/Jannat24/uncovering_deepfake_image) <br/>

## BibTeX

```
@InProceedings{uncovering-deepfake,
author = {Syeda Jannatul Naim and Sarker Tanveer Ahmed Rumee},
title = {Uncovering DeepFake Images for Identifying Source-images},
booktitle = {Proceedings of the 27th International Conference on Computer and Information Technology (\textsc{ICCIT}'24)}, 
year = {2024}, 
owner = {Syeda Jannatul Naim}, 
address = {Cox’s Bazar, Bangladesh} }

```


