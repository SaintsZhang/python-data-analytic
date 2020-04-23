import glob
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