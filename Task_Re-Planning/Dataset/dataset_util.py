# 从之前场景图数据集的原始文件annotations_txt中提取部分可用于任务重的数据
import os,sys
import shutil
import re


def annoToraw():
	path_t = r'/home/lx/TaskPlaning/datasets/annotations_txt'  #按照自己的路径进行修改
	txt_list = os.listdir(path_t)
	# print(txt_list)
	# print(txt_list)
	# print(len(txt_list))
	i = 0
	for txt_name in txt_list:
		txt_name = os.path.join(path_t, txt_name)
		with open(txt_name) as f:
			listobject = []
			for f1 in f.readlines():
				f1 = f1.replace("\n", "").strip().split(' ')
				listobject.append(f1[0])
		f.close()
		
		if ('manipulator' in listobject and 'bowl' in listobject) or ('manipulator' in listobject and 'plate' in listobject):
	#         if 'manipulator' in listobject and 'bowl' in listobject and 'plate' in listobject and 'apple' in listobject and 'banana' in listobject and 'orange' in listobject and 'cup' in listobject:
			shutil.copy(txt_name,r'/home/lx/TaskPlaning/datasets/raw_data/')

# 根据提取到的数据将对应的图像复制到taskplaning文件夹——用于数据集标注
def picTopic():
	path_t = r'/home/lx/TaskPlaning/datasets/'
	txt_list = os.listdir(path_t + '/raw_data/')
	for txt_name in txt_list:
		s = re.findall("\d+",txt_name)[0]
	#     print('/home/lx/DRNet/datasets/data_yolo/JPEGImages/' + s+'.jpg')
		shutil.copy('/home/lx/DRNet/datasets/data_yolo/JPEGImages/' + s+'.jpg' , path_t+ 'raw_picture')


def rename():
    path_t = r'/home/lx/TaskPlaning/datasets/5pour_water_from_cup_into_bowl_annotations/'
    txt_list = os.listdir(path_t + '/raw_data/')
    n = 271
    for txt_name in txt_list:
        s = re.findall("\d+",txt_name)[0]
        oldname = path_t + '/raw_data/'+txt_name
        newname = path_t + '/raw_data1/'+str(n)+'.txt'
        oldname_anno = path_t + '/task_action/'+txt_name
        newname_anno = path_t + '/task_action1/'+str(n)+'.txt'
        oldname_pic = path_t + '/raw_picture/'+ s +'.jpg'
        newname_pic = path_t + '/raw_picture1/'+str(n)+'.jpg'
        os.rename(oldname,newname)
        os.rename(oldname_anno,newname_anno)
        os.rename(oldname_pic,newname_pic)
        n += 1
        
        
#         print(s)
#     print('/home/lx/DRNet/datasets/data_yolo/JPEGImages/' + s+'.jpg')
#         shutil.copy('/home/lx/DRNet/datasets/data_yolo/JPEGImages/' + s+'.jpg' , path_t+ '/raw_picture')



