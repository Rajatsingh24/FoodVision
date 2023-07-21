# Food Vision 
> ***TO visit the model [click here](https://huggingface.co/spaces/rajatsingh0702/FoodVision)***

A model which can classify 101 different food images.
In this repository we are going to fine-tune a pre-trained model(EfficientNet-B2) on Food101 Dataset. [Click here](https://github.com/Rajatsingh24/FoodVision/blob/main/class_names.txt) to open text file containing 101 food labels the are present in model.

***Remark :*** To train the model run train.py script.
### File :
* *data_setup.py* - Contains functions (create_dataloaders()) which is used to create DataLoaders
* *engin.py* - Contains functions (train_step(),test_step(),train()) to train the data.
* *model_builder.py* - Contains functions (model_build()) to build the model.
* *pridiction.py* - Contaions funtions (pred_and_plot_image()) to get the pridiction
* *train.py* - A script to connect different sub-files.
* *utils.py* - Contains function (save_model()) to save the model.


## Website ScreenShot: 

![](https://www.imgbly.com/ib/06YHx30TVx.png)
![](https://www.imgbly.com/ib/26AXWo07IX.png)

## Created BY :

### *Rajat Singh*

[Website-link](https://huggingface.co/spaces/rajatsingh0702/FoodVision)

[GitHub](https://github.com/Rajatsingh24 "gitHub link")