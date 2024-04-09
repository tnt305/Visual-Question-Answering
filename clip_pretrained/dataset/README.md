## Download dataset
**Note:** If you can't download using gdown due to limited number of downloads, please download it manually and upload it to your drive, then copy it from the drive to colab.
```python
from google.colab import drive

drive.mount('/content/drive')
!cp /path/to/dataset/on/your/drive .
```

You can download the dataset via this [Google Drive](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view?usp=sharing) then unzip using `!unzip -q vqa_coco_dataset.zip`

A simple look for our dataset
<p> align="center">
 <img src="fig/dataset.png" width="800">
</p>