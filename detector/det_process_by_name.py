import numpy as np
import argparse

# 检测结果后处理

def back_process(vd_name):
    txt_path = './result/img/%s/det/det.txt'%vd_name
    try:
        a = np.loadtxt(txt_path,delimiter=',')
    except:
        print("no %s file"%vd_name)
    
    # index_small = np.where(a[:,4]*a[:,5]<1000)[0]
    index_small = np.where((a[:,4]*a[:,5]<1000)&(a[:,3]>1080//2))[0]

    save_det = np.delete(a,index_small,axis=0)

    np.savetxt('./result/img/%s/det/det.txt'%vd_name,save_det,fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")


    # txt_path = './result/img/b4/det/det.txt'
    # try:
    #     a = np.loadtxt(txt_path,delimiter=',')
    # except:
    #     a = np.loadtxt(txt_path)

    # index_near = np.where(a[:,2]<100)[0]
    # area = np.sort(a[:,4]*a[:,5])[-20:]
    # width = np.sort(a[:,4])[-20:]
    # length = np.sort(a[:,4])[-20:]
    # #print('length',length)
    # length_index = np.where(a[:,5] > 600)[0]
    # width_index = np.where(a[:,4] > 300)[0]
    # frame_length = a[length_index,0]
    # #print('fra  asd',frame_length)
    # index_large = np.where(a[:,4]*a[:,5]>100000)[0]
    # #print(index_near,len(index_near) )
    # #print('\n',index_large,len(index_large))
    # delete_index = np.concatenate((index_near,index_large,length_index,width_index),axis=0)
    # save_list = np.delete(a,delete_index,axis=0)
    # #print(np.shape(save_list))
    # #print(width)
    # np.savetxt('./result/img/b4/det/det.txt',save_list,fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='file name')

    opt = parser.parse_args()

    back_process(opt.file)