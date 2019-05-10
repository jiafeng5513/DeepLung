# DeepLung:
#### Start-up operation guide
1. 下载数据集
2. 运行`./prepare.py`
3. 训练检测器.检测器在detector中,通过运行`./run_trainging_detector.sh`开始训练<br>
4. 测试检测器.通过运行`./run_test_detector.sh`进行
5. 分类器需要的输入尺度和检测器不同,需要先进行数据准备,通过`./nodcls/Create_crop_v3.py`生成检测器的输入数据.<br>
6. 训练分类器.分类器在nodcls中,通过运行`./run_training_classifier.sh`开始训练
7. 
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

1. After training and test are done, use the `./evaluationScript/frocwrtdetpepchluna16.py` to validate the epoch used for test. 
2. After that, collect all the 10 folds' prediction, use `./evaluationScript/noduleCADEvaluationLUNA16.py` to get the FROC for all 10 folds. You can directly run `noduleCADEvaluationLUNA16.py`, and get the performance in the paper.

3.  For system's classification, that is classification based on detection. 
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
   2. `./nodcls/data/extclsshpinfo.py` is used to extract nodule labels.
   3. `./nodcls/data/humanperformance.py` is used to get the performance of doctors. 
   4. `./nodcls/data/dimcls.py` is used to get the classification based on diameter.
   5. `./nodcls/data/nodclsgbt.py` is used to get the performance based on GBM, nodule diameter and nodule pixel.
   6. `./nodcls/data/pthumanperformance.py` is used for patient-level diagnosis performance. 
   7. `./nodcls/data/kappatest.py` is used for kappa value calculation in the paper.
   8. For classification using DPN, use the code in `./nodcls/main_nodcls.py`. 
   9. Use the `./nodcls/testdet2cls.py` to test the trained model. You may revise the code a little bit for different test settings.
6. `./config_training.py`中包含了训练所需的参数,主要是LUNA16数据集的路径.
7. `./prepare.py`是数据预处理脚本,预处理产生的数据也会存储在`./config_training.py`中指定的文件夹中.

Doctor's annotation for each nodule in LIDC-IDRI is in ./nodcls/annotationdetclssgm_doctor.csv
#### The performances on each fold are (these results are in the supplement)

|          |Deep 3D Res18|Deep 3D DPN26|
|:--------:|:-----------:|:-----------:|
|Fold 0    |       0.8610|	     0.8750|
|Fold 1    |       0.8538|	     0.8783|
|Fold 2    |       0.7902|       0.8170|
|Fold 3    |       0.7863|       0.7731|
|Fold 4    |       0.8795|	     0.8850|
|Fold 5    |       0.8360|  	 0.8095|
|Fold 6    |       0.8959|  	 0.8649|
|Fold 7    |       0.8700|       0.8816|
|Fold 8    |       0.8886|	     0.8668|
|Fold 9    |       0.8041|    	 0.8122|

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

