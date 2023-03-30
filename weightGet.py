import wget
import os

def download(url, out_path='.'):
    wget.download(url, out=out_path)

def pathCheck(path):
    if not os.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    url = "https://github.com/Ted521/RootLab_Welding/releases/download/weightFile/latest.pth"
    path = "./RootLab_Welding/data/work"
    pathCheck(path)
    fname = "./RootLab_Welding/data/work/latest.pth"
    download(url, out_path=fname)