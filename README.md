# Image-Segmentation
Image-Segmentation on cityscapes-dataset 
+ Dataset overview : https://www.cityscapes-dataset.com/dataset-overview

![cityscapes](https://user-images.githubusercontent.com/38074499/62729743-cc545000-ba48-11e9-84f3-9199ee31eeb9.png)

+ For dowload dataset see: dowload_data.ipynb : 
  + First, you need to create an account in the web page. You will use your username and password in the first line of the script to login to the page.
  ![image](https://user-images.githubusercontent.com/38074499/62730013-58ff0e00-ba49-11e9-913f-3ddbe35ace49.png)
  
  + In the first line, put your username and password. This will login with your credentials and keep the associated cookies.
  + In the second line, you need to provide the packageID paramater and it downloads the file.
  + packageIDs map like this in the website:
    + 1 -> gtFine_trainvaltest.zip (241MB) 
    + 2 -> gtCoarse.zip (1.3GB) 
    + 3 -> leftImg8bit_trainvaltest.zip (11GB) 
    + 4 -> leftImg8bit_trainextra.zip (44GB) 
    + 8 -> camera_trainvaltest.zip (2MB) 
    + 9 -> camera_trainextra.zip (8MB) 
    + 10 -> vehicle_trainvaltest.zip (2MB) 
    + 11 -> vehicle_trainextra.zip (7MB) 
    + 12 -> leftImg8bit_demoVideo.zip (6.6GB) 
    + 28 -> gtBbox_cityPersons_trainval.zip (2.2MB)
 + See original article at : https://github.com/cemsaz/city-scapes-script
 
 ## Model overview:
 + FCN
  ![image](https://user-images.githubusercontent.com/38074499/62730581-741e4d80-ba4a-11e9-9ef4-f35350a2a72f.png)
  
 + Unet
 ![image](https://user-images.githubusercontent.com/38074499/62730618-89937780-ba4a-11e9-8157-c704ccdb583a.png)
