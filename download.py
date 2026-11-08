"""Download MNIST training files from the CVDF mirror. Source attribution in README."""
from pathlib import Path
import urllib.request
BASE='https://storage.googleapis.com/cvdf-datasets/mnist/'
def main():
 p=Path('data');p.mkdir(exist_ok=True)
 for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz']:
  if not (p/name).exists():urllib.request.urlretrieve(BASE+name,p/name)
 print('MNIST training files ready. Source: https://github.com/cvdfoundation/mnist')
if __name__=='__main__':main()
