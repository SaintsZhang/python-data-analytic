import glob
import imageio
import os

class fileUtil:
    def __init__(self):
        pass
    def __repContent(self,file,searchExp,replaceExp):
        fin = open(file,'rt')
        data = fin.read()
        data = data.replace(searchExp,replaceExp)
        fin = open(file, "wt")
        fin.write(data)
        fin.close()
    def replaceFiles(self, dir,searchExp,replaceExp):
        files = glob.glob(dir)
        for f in files:
            print(f)
            self.__repContent(f, searchExp, replaceExp)  

class imageUtil:
    def __init__(self):
        pass
    def create_gif(self,image_list,gif_name,duration = 1.0):
        '''
        to create a gif 
        :param image_list:
        :param gif_name:
        :param duration:
        :return:
        '''
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        imageio.mimsave(gif_name, frames,'GIF', duration = duration)
        return
    def get_all_file(self,file_dir,tail_list=('.jpg','.png','jpeg')):
        file_list = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                for tail in tail_list:
                    if file.endswith(tail):
                       file_list.append(os.path.join(root,file)) 
                       break
        return file_list              