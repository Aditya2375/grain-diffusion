"""Fixed shared-noise probes. Noise MSE, not generated image quality."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from train_guided import Guided

def main():
 torch.set_num_threads(2);base=json.loads(Path('2-model.json').read_text());new=json.loads(Path('guided-model.json').read_text());nets=[]
 for m in [base,new]:
  n=Guided();n.load_state_dict({k:torch.tensor(v) for k,v in m['weights'].items()});n.eval();nets.append(n)
 raw=gzip.decompress(Path('data/train-images-idx3-ubyte.gz').read_bytes());lab=gzip.decompress(Path('data/train-labels-idx1-ubyte.gz').read_bytes());x=torch.tensor(np.frombuffer(raw[16:],np.uint8).copy().reshape(-1,1,28,28)).float()/255;x=nn.functional.avg_pool2d(x,2).reshape(-1,196)*2-1;y=torch.tensor(np.frombuffer(lab[8:],np.uint8).copy()).long();g=torch.Generator().manual_seed(1281);ids=torch.randint(5000,(2048,),generator=g)+55000;t=torch.randint(64,(2048,),generator=g);noise=torch.randn(2048,196,generator=g);a=torch.tensor(new['alpha_bar'],dtype=torch.float32)[t,None];xx=a.sqrt()*x[ids]+(1-a).sqrt()*noise
 with torch.no_grad():losses=[(n(xx,t,y[ids])-noise).square().mean(1).numpy() for n in nets]
 # Resample source-image clusters, not noise probes independently (sampled image ids can repeat).
 unique=np.unique(ids.numpy());sums=np.array([np.sum((losses[0]-losses[1])[ids.numpy()==i]) for i in unique]);counts=np.array([np.sum(ids.numpy()==i) for i in unique]);r=np.random.default_rng(321);bs=[]
 for _ in range(3000):
  k=r.integers(len(unique),size=len(unique));bs.append(float(sums[k].sum()/counts[k].sum()))
 report=dict(probes=2048,unique_images=len(unique),seed=1281,old_mse=float(losses[0].mean()),new_mse=float(losses[1].mean()),reduction=float((losses[0]-losses[1]).mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),bootstrap_seed=321,resamples=3000,scope='Existing last5000 validation partition, fixed shared-noise probes, source-image cluster bootstrap. Not untouched test or image quality. No checkpoint selection.')
 Path('noise-audit.json').write_text(json.dumps(report,separators=(',',':')));np.savez_compressed('noise-probes.npz',old=losses[0],new=losses[1],ids=ids.numpy());print(report)
if __name__=='__main__':main()
