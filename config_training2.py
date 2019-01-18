config = {'train_data_path':['/home/RAID1/DataSet/LUNA16/subset0/',
                             '/home/RAID1/DataSet/LUNA16/subset1/',
                             '/home/RAID1/DataSet/LUNA16/subset9/',
                             '/home/RAID1/DataSet/LUNA16/subset3/',
                             '/home/RAID1/DataSet/LUNA16/subset4/',
                             '/home/RAID1/DataSet/LUNA16/subset5/',
                             '/home/RAID1/DataSet/LUNA16/subset6/',
                             '/home/RAID1/DataSet/LUNA16/subset7/',
                             '/home/RAID1/DataSet/LUNA16/subset8/'],
          'val_data_path':['/home/RAID1/DataSet/LUNA16/subset2/'], 
          'test_data_path':['/home/RAID1/DataSet/LUNA16/subset2/'], 
          
          'train_preprocess_result_path':'/home/RAID1/DataSet/LUNA16/preprocess/',
          'val_preprocess_result_path':'/home/RAID1/DataSet/LUNA16/preprocess/',  
          'test_preprocess_result_path':'/home/RAID1/DataSet/LUNA16/preprocess/',
          
          'train_annos_path':'/home/RAID1/DataSet/LUNA16/CSVFILES/annotations.csv',
          'val_annos_path':'/home/RAID1/DataSet/LUNA16/CSVFILES/annotations.csv',
          'test_annos_path':'/home/RAID1/DataSet/LUNA16/CSVFILES/annotations.csv',

          'black_list':['1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286',
                        '1.3.6.1.4.1.14519.5.2.1.6279.6001.132817748896065918417924920957'],
          
          'preprocessing_backend':'python',
         } 