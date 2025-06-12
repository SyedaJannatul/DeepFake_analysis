# Uncovering DeepFake Images for Identifying Source-images
## Paper:
**Abstract**: Owing to the swift advancement of face swapping technology utilized in creating ”Deepfakes” and the increasing public awareness regarding multimedia security, numerous studies have been undertaken in recent years to devise various methods for detecting deepfakes. This paper presents a framework aimed at enhancing clarity in deepfake detection techniques through the reconstruction of source images from a provided deepfake image. Our focus is on developing a system that not only addresses the inquiry, ”Is this image a DeepFake or not?” but also provides insights into the question, ”If the image is a DeepFake, what are the source images?” An extensive experiment was conducted on our hypothesis, yielding significant results from the exploration.
<br/><br/>
**Keywords:** DeepFake, Faceswap, DeepFake image analysis, DeepFake source images.
<br/><br/>
[Preprint](https://www.researchgate.net/publication/387024703_Uncovering_DeepFake_Images_for_Identifying_Source-images)<br/>

## Dataset:
[**DeepFake(Faceswapped Images)**](https://www.kaggle.com/datasets/syedajannatulnaim/deepfakeface-swapped-images-using-ffhq-dataset)<br/>
[**Background removed images**](https://www.kaggle.com/datasets/syedajannatulnaim/background-removed-images-of-ffhq-dataset?select=flickr_remb)<br/>
[**Segmented Face images**](https://www.kaggle.com/datasets/syedajannatulnaim/background-removed-images-of-ffhq-dataset?select=segmented_face)<br/>

## Instructions:
```
git clone https://github.com/CompVis/taming-transformers.git
```

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
@INPROCEEDINGS{11021847,
  author={Naim, Syeda Jannatul and Rumee, Sarker Tanveer Ahmed},
  booktitle={2024 27th International Conference on Computer and Information Technology (ICCIT)}, 
  title={Uncovering DeepFake Images for Identifying Source-images}, 
  year={2024},
  pages={2629-2634},
  keywords={Deepfakes;Image analysis;Computer architecture;Security;Information technology;Image reconstruction;Faces;DeepFake;Faceswap;DeepFake image analysis;DeepFake source images},
  doi={10.1109/ICCIT64611.2024.11021847}}

```


