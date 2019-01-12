# DeepLung:
#### Note
Forked From [wentaozhu/DeepLung](https://github.com/wentaozhu/DeepLung) and edited by [jiafeng5513](https://github.com/jiafeng5513).

#### Citation:
Zhu, Wentao, Chaochun Liu, Wei Fan, and Xiaohui Xie. "DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification." IEEE WACV, 2018.

#### Dependecies: 
Ubuntu 14.04, python 2.7, CUDA 8.0, cudnn 5.1, h5py (2.6.0), SimpleITK (0.10.0), numpy (1.11.3), nvidia-ml-py (7.352.0), matplotlib (2.0.0), scikit-image (0.12.3), scipy (0.18.1), pyparsing (2.1.4), pytorch (0.1.10+ac9245a) (anaconda is recommended)

#### Data Set:
Download LUNA16 dataset from https://luna16.grand-challenge.org/data/
Download LIDC-IDRI dataset from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

#### Run:
1. For preprocessing, run `./DeepLung/prepare.py`.
2. The parameters for `prepare.py` is in config_training.py. 
   1. `*_data_path` is the unzip raw data path for LUNA16.
   2. `*_preprocess_result_path` is the save path for the preprocessing.
   3. `*_annos_path` is the path for annotations. 
   4. `*_segment` is the path for LUNA16 segmentation, which can be downloaded from LUNA16 website.
3. Use `run_training.sh` to train the detector. You can use the resnet or dual path net model by revising `--model` attribute. 
4. After training and test are done, use the `./evaluationScript/frocwrtdetpepchluna16.py` to validate the epoch used for test. 
5. After that, collect all the 10 folds' prediction, use `./evaluationScript/noduleCADEvaluationLUNA16.py` to get the FROC for all 10 folds. You can directly run `noduleCADEvaluationLUNA16.py`, and get the performance in the paper.
6.  The trained model is in `./detector/dpnmodel/` or `./detector/resmodel/`
7.  For nodule classification, first clean the data from LIDC-IDRI. Use the `./data/extclsshpinfo.py` to extract nodule labels. `humanperformance.py` is used to get the performance of doctors. 
8.  `dimcls.py` is used to get the classification based on diameter. 
9.  `nodclsgbt.py` is used to get the performance based on GBM, nodule diameter and nodule pixel.
10. `pthumanperformance.py` is used for patient-level diagnosis performance. 
11. `kappatest.py` is used for kappa value calculation in the paper.
12. For classification using DPN, use the code in `main_nodcls.py`. 
13. Use the `testdet2cls.py` to test the trained model. You may revise the code a little bit for different test settings.
14. For system's classification, that is classification based on detection. 
    1.  First, use the detection's test script in the `run_training.sh` to get the detected nodules for training CTs.
    2.  Use the `det2cls.py` to train the model. 
    3.  And use the `testdet2cls.py` to test the trained model. 
    4.  You may revise the code a little bit for different test settings.
#### Files description:
1. `./DeepLungDetectionDemo/`文件夹中存储的是一个测试demo.
2. `./detector/`文件夹中是检测器的相关代码,
   1. 入口在`./detector/main.py`.
   2. 可以通过`./run_training.sh`启动训练,`--model`参数用于选择resnet 和 dual path net
   3. `./detector/dpnmodel/` 和 `./detector/resmodel/`中存储训练好的模型
3. `./DOCs/`文件夹中存储文档,包含环境说明文档和论文.
4. `./evaluationScripts/`文件夹中含有LUNA16提供的评估脚本,具体情况请看`./evaluationScripts/README.md`
   1. 训练和测试结束后,`./evaluationScript/frocwrtdetpepchluna16.py`可以用来验证.
   2. `./evaluationScript/noduleCADEvaluationLUNA16.py`用来生成论文中的一些图.
5. `./nodcls/`文件夹中是分类器的相关代码.
   1. clean the data from LIDC-IDRI.
   2. Use the `./nodcls/data/extclsshpinfo.py` to extract nodule labels.
   3. `./nodcls/data/dimcls.py` is used to get the classification based on diameter.
   4. `./nodcls/data/nodclsgbt.py` is used to get the performance based on GBM, nodule diameter and nodule pixel.
   5. `./nodcls/data/pthumanperformance.py` is used for patient-level diagnosis performance. 
   6. `./nodcls/data/kappatest.py` is used for kappa value calculation in the paper.
   7. For classification using DPN, use the code in `./nodcls/main_nodcls.py`. 
   8. Use the `./nodcls/testdet2cls.py` to test the trained model. You may revise the code a little bit for different test settings.
6. `./config_training.py`中包含了训练所需的参数,主要是LUNA16数据集的路径.
7. `./prepare.py`是数据预处理脚本,预处理产生的数据也会存储在`./config_training.py`中指定的文件夹中.

Doctor's annotation for each nodule in LIDC-IDRI is in ./nodcls/annotationdetclssgm_doctor.csv
#### The performances on each fold are (these results are in the supplement)

|          |Deep 3D Res18|Deep 3D DPN26|
|:--------:|:-----------:|:------------:|
|Fold 0    |       0.8610|	      0.8750|
|Fold 1    |       0.8538|	      0.8783|
|Fold 2    |       0.7902|        0.8170|
|Fold 3    |       0.7863|        0.7731|
|Fold 4    |       0.8795|	      0.8850|
|Fold 5    |       0.8360|  	  0.8095|
|Fold 6    |       0.8959|  	  0.8649|
|Fold 7    |       0.8700|        0.8816|
|Fold 8    |       0.8886|	      0.8668|
|Fold 9    |       0.8041|    	  0.8122|

#### The performances on each average false positives in FROC compared with other approaches (these results are in the supplement)

|                | 0.125|  0.25|   0.5|     1|     2|     4|     8|  FROC|
|:--------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|DIAG_ConvNet    | 0.692| 0.771| 0.809| 0.863| 0.895| 0.914| 0.923| 0.838|
|ZENT            | 0.661| 0.724| 0.779| 0.831| 0.872| 0.892| 0.915| 0.811|
|Aidence         | 0.601| 0.712| 0.783| 0.845| 0.885| 0.908| 0.917| 0.807|
|MOT_M5Lv1       | 0.597| 0.670| 0.718| 0.759| 0.788| 0.816| 0.843| 0.742|
|VisiaCTLung     | 0.577| 0.644| 0.697| 0.739| 0.769| 0.788| 0.793| 0.715|
|Etrocad         | 0.250| 0.522| 0.651| 0.752| 0.811| 0.856| 0.887| 0.676|
|Dou et al 2017  | 0.659| 0.745| 0.819| 0.865| 0.906| 0.933| 0.946| 0.839|
|3D RES          | 0.662| 0.746| 0.815| 0.864| 0.902| 0.918| 0.932| 0.834|
|3D DPN          | 0.692| 0.769| 0.824| 0.865| 0.893| 0.917| 0.933| 0.842|


#### Aboub
1. Feel free to ask any questions. Wentao Zhu, wentaozhu1991@gmail.com
2. LIDC-IDRI nodule size report downloaded from 
http://www.via.cornell.edu/lidc/list3.2.csv is in /nodcls/data/list3.2.csv


#### Note by jiafeng5513
1. `./DOCs/environment.md` is the environment settings.
2. `frocwrtdetpepchluna16.py`中有一些变量没有初始化.
3. 