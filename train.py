"""Conditional pixel DDPM on downsampled MNIST, trained from scratch on CPU."""
import json,gzip,hashlib,math
from pathlib import Path
import numpy as np,torch
from torch import nn
class Net(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(196+16+10,256);self.fc2=nn.Linear(256,256);self.fc3=nn.Linear(256,196)
 def forward(self,x,t,y):
  freq=torch.exp(-math.log(10000)*torch.arange(8)/7);a=t[:,None].float()*freq[None,:];emb=torch.cat([a.sin(),a.cos()],1);h=torch.cat([x,emb,nn.functional.one_hot(y,10).float()],1);return self.fc3(nn.functional.silu(self.fc2(nn.functional.silu(self.fc1(h)))))
def schedule(T=64):
 s=.008;a=np.cos(((np.arange(T+1)/T+s)/(1+s))*np.pi/2)**2;a=a/a[0];beta=np.clip(1-a[1:]/a[:-1],0,.999);return beta,np.cumprod(1-beta)
def main():
 torch.set_num_threads(2);torch.manual_seed(431);r=Path('data');raw=gzip.decompress((r/'train-images-idx3-ubyte.gz').read_bytes());labels=gzip.decompress((r/'train-labels-idx1-ubyte.gz').read_bytes());x=torch.tensor(np.frombuffer(raw[16:],np.uint8).copy().reshape(-1,1,28,28)).float()/255;x=nn.functional.avg_pool2d(x,2).reshape(-1,196)*2-1;y=torch.tensor(np.frombuffer(labels[8:],np.uint8).copy()).long();train=x[:55000];val=x[55000:];ty=y[:55000];vy=y[55000:];beta,abar=schedule();ab=torch.tensor(abar,dtype=torch.float32);net=Net();optim=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001);g=torch.Generator().manual_seed(431);vg=torch.Generator().manual_seed(913);history=[]
 def evaluate():
  net.eval();losses=[]
  with torch.no_grad():
   for _ in range(12):
    ids=torch.randint(len(val),(256,),generator=vg);t=torch.randint(64,(256,),generator=vg);noise=torch.randn(256,196,generator=vg);a=ab[t,None];noisy=a.sqrt()*val[ids]+(1-a).sqrt()*noise;losses.append((net(noisy,t,vy[ids])-noise).square().mean().item())
  net.train();return float(np.mean(losses))
 history.append(dict(step=0,validation=evaluate()));print(history[-1],flush=True);sums=[]
 for step in range(1,30001):
  ids=torch.randint(len(train),(128,),generator=g);t=torch.randint(64,(128,),generator=g);noise=torch.randn(128,196,generator=g);a=ab[t,None];noisy=a.sqrt()*train[ids]+(1-a).sqrt()*noise;loss=(net(noisy,t,ty[ids])-noise).square().mean();optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optim.step();sums.append(loss.item())
  if step%500==0:history.append(dict(step=step,train=float(np.mean(sums)),validation=evaluate()));sums=[];print(history[-1],flush=True)
 net.eval();fixtures=[]
 with torch.no_grad():
  for label,t in [(0,63),(3,24),(9,0)]:
   a=torch.linspace(-1,1,196)[None,:];out=net(a,torch.tensor([t]),torch.tensor([label]))[0].tolist();fixtures.append(dict(input=a[0].tolist(),t=t,label=label,output=out))
 payload=dict(version=1,config=dict(size=14,steps=64,hidden=256,embedding=16,classes=10),weights={k:v.tolist() for k,v in net.state_dict().items()},beta=beta.tolist(),alpha_bar=abar.tolist(),training=dict(seed=431,updates=30000,batch=128,learning_rate=.0007,train_images=55000,validation_images=5000,history=history,parameters=sum(p.numel() for p in net.parameters()),sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.gz')}),fixtures=fixtures)
 Path('1-model.json').write_text(json.dumps(payload,separators=(',',':')));print('done',flush=True)
if __name__=='__main__':main()
