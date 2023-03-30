import wget

def download(url, out_path='.'):
    wget.download(url, out=out_path)

if __name__ == "__main__":
    url = "https://github.com/Ted521/RootLab_Welding/releases/download/weightFile/latest.pth"
    download(url, out_path="./RootLab_Welding/data/work")