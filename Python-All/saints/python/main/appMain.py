from saints.python.classes.szutils import fileUtil as fu

def main():
    c = fu()
    c.replaceFiles(r"c:\saints\*.plsql", "utl_file.fopen","fopen")
if __name__=="__main__":
    main()