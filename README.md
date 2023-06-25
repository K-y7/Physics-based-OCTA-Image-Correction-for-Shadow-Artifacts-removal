# Removal-of-Shadow-Artifacts-in-OCTA-Retinal-Images
## Intruduction
We propose a deep learning-based method that can directly detect and remove shadowing artifacts in retinal OCTA images without damaging the retinal structure. Experiments show that the average non-luminance structure similarity before and after artifact removal reached 0.94. The retinal images with shadowing artifacts have high NPA values, which can be reduced to normal range after processing by this method, improving the reliability and accuracy of the medical indicator NPA.<br>
This method was developed by Kang Wang from TGU-UOW laboratory based on SpA-Former.<br>
Office Website：[TGU-UOW](http://tgu-uow.gitee.io/)
# Usage
·pytorch 1.8.1<br>
·python 3.8
## Train
Modify the config.yml to set your parameters and run:<br>
```
python train.py
``` 

## Test
Although the image size we trained is 256*256, the predicted image can be any size, you just need to modify the length and width in config.yml. <br>
```
python predict.py --config <path_to_config.yml_in_the_out_dir> --test_dir <path_to_a_directory_stored_test_data> --out_dir <path_to_an_output_directory> --pretrained <path_to_a_pretrained_model> --cuda
```
## Pretrained model
Download the pretrained model shadow artifact-removal [Baidu Drive](https://pan.baidu.com/s/1Vh4FiW_cUK_0mXauz1mZsA) extract code：epzo  
There're my pre-trained models on OCTA images<br>
![Result](https://github.com/K-y7/Removal-of-shadow-artifacts-in-OCTA-retinal-images/blob/master/imgs/result.png)
## Contact
## Acknowledgment
The code is based on https://github.com/zhangbaijin/SpA-Former-shadow-removal
