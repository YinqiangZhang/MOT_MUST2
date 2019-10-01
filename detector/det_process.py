import numpy as np
# 检测结果后处理

files = ['b2','b3','b4','b5']
for i in files:
    txt_path = './result/img/%s/det/det.txt'%i
    try:
        a = np.loadtxt(txt_path,delimiter=',')
    except:
        print("no %s file"%i)
        continue
        
    index_small = np.where(a[:,4]*a[:,5]<1000)[0]
    save_list = np.delete(a,index_small,axis=0)
    np.savetxt('./result/img/%s/det/det.txt'%i,save_list,fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")

txt_path = './result/img/b4/det/det.txt'
try:
    a = np.loadtxt(txt_path,delimiter=',')
except:
    a = np.loadtxt(txt_path)
#print(np.shape(a))

index_near = np.where(a[:,2]<100)[0]
area = np.sort(a[:,4]*a[:,5])[-20:]
width = np.sort(a[:,4])[-20:]
length = np.sort(a[:,4])[-20:]
#print('length',length)
length_index = np.where(a[:,5] > 600)[0]
width_index = np.where(a[:,4] > 300)[0]
frame_length = a[length_index,0]
#print('fra  asd',frame_length)
index_large = np.where(a[:,4]*a[:,5]>100000)[0]
#print(index_near,len(index_near) )
#print('\n',index_large,len(index_large))
delete_index = np.concatenate((index_near,index_large,length_index,width_index),axis=0)
save_list = np.delete(a,delete_index,axis=0)
#print(np.shape(save_list))
#print(width)

np.savetxt('./result/img/b4/det/det.txt',save_list,fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")