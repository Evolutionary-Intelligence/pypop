Code in this directory is referenced from: \
Zhong et al., 2022. \
Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon. \
IEEE Conference on Computer Vision and Pattern Recognition.\
https://arxiv.org/abs/2203.03818 \
The code is located at:\
https://github.com/hncszyq/ShadowAttack \

**Datasets and trained models**  
   You should first download [the LISA and GTSRB datasets](https://drive.google.com/file/d/1Du8egeUG6XgAVf-h9IcxRz5gZvs7_Ldq/view?usp=sharing) and [our trained models](https://drive.google.com/file/d/1C0k77EeZrByBUdv36IxS9PiLUvZXRr24/view?usp=sharing) and place them in dataset/ and model/, respectively.  

**Requirements:**  
   ```text
   python >= 3.8.11
   pytorch >= 1.9.0
   torchvision >= 0.10.0
   shapely
   opencv-python 
   ```
   To run in an environment without cuda enabled, change `"device": "cuda:0"` to `"device": "cpu"` in params.json.
   
- **Example 1: show help message.**
   ```shell
   $ python3 shadow_attack.py --help
   ```
   ```text
   usage: shadow_attack.py [-h] [--shadow_level SHADOW_LEVEL] [--attack_db ATTACK_DB] [--attack_type ATTACK_TYPE] [--image_path IMAGE_PATH] [--mask_path MASK_PATH] [--image_label IMAGE_LABEL] [--polygon POLYGON] [--n_try N_TRY] [--target_model TARGET_MODEL]

   optional arguments:
   -h, --help           show this help message and exit
   --shadow_level       shadow coefficient k
   --attack_db          the target dataset should be specified for a digital attack
   --attack_type        digital attack or physical attack
   --image_path         the file path to the target image should be specified for a physical attack
   --mask_path          the file path to the mask should be specified for a physical attack
   --image_label        the ground truth should be specified for a physical attack
   --polygon            The number of sides of polygon P.
   --n_try              n-random-start strategy: retry n times
   --target_model       attack normal model or robust model
   ```
 
 - **Example 2: physical attack:**  
   The following shell will launch our physical attack. The generated images will be saved as ./tmp/adv_img.png.
   ```shell
   $ python3 shadow_attack.py --shadow_level 0.43 --attack_db GTSRB --attack_type physical --image_path ./tmp/gtsrb_30.png --mask_path ./tmp/gtsrb_30_mask.png --image_label 1 
   ```
   ```text
   iteration: 1 0.7416303753852844
   iteration: 2 0.7226707339286804
   iteration: 3 0.7226707339286804
   iteration: 4 0.7226707339286804
   iteration: 5 0.6017401814460754
   ...
   iteration: 198 0.07176369428634644
   iteration: 199 0.07176369428634644
   iteration: 200 0.07176369428634644
   Best solution: 0.9282363057136536 succeed
   Correct: False Predict: 40 Confidence: 90.27248024940491%
   Attack succeed! Try to implement it in the real world.
   ```
 
 - **Example 3: Change the number of sides of polygon P, e.g., 4:**
   ```shell
   $ python3 shadow_attack.py --attack_db LISA --polygon 4