"""Fixed-budget class-dropout fine-tuning from the original local DDPM."""
import json,math,gzip,hashlib,time
from pathlib import Path
import torch,numpy as np
from torch import nn
from train import Net
class Guided(Net):
 def forward(self,x,t,y):
  freq=torch.exp(-math.log(10000)*torch.arange(8)/7);a=t[:,None].float()*freq[None,:];emb=torch.cat([a.sin(),a.cos()],1);condition=nn.functional.one_hot(y.clamp(min=0),10).float()*(y[:,None]>=0);h=torch.cat([x,emb,condition],1);return self.fc3(nn.functional.silu(self.fc2(nn.functional.silu(self.fc1(h)))))
def main():
 torch.set_num_threads(2);torch.manual_seed(731);base=json.loads(Path('2-model.json').read_text());root=Path('data')
 for n,h in base['training']['sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
 raw=gzip.decompress((root/'train-images-idx3-ubyte.gz').read_bytes());lab=gzip.decompress((root/'train-labels-idx1-ubyte.gz').read_bytes());x=torch.tensor(np.frombuffer(raw[16:],np.uint8).copy().reshape(-1,1,28,28)).float()/255;x=nn.functional.avg_pool2d(x,2).reshape(-1,196)*2-1;y=torch.tensor(np.frombuffer(lab[8:],np.uint8).copy()).long();ab=torch.tensor(base['alpha_bar'],dtype=torch.float32);net=Guided();net.load_state_dict({k:torch.tensor(v) for k,v in base['weights'].items()});o=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001);g=torch.Generator().manual_seed(731);hist=[];ls=[]
 # Fixed validation probes, same last5000 heldout partition as v1, not untouched test.
 vg=torch.Generator().manual_seed(1281);ids=torch.randint(5000,(2048,),generator=vg)+55000;vt=torch.randint(64,(2048,),generator=vg);vn=torch.randn(2048,196,generator=vg);va=ab[vt,None];vx=va.sqrt()*x[ids]+(1-va).sqrt()*vn
 def ev():
  with torch.no_grad():
   return dict(conditional=float((net(vx,vt,y[ids])-vn).square().mean()),unconditional=float((net(vx,vt,torch.full_like(y[ids],-1))-vn).square().mean()))
 before=ev();print('before',before,flush=True)
 for step in range(1,12001):
  ids2=torch.randint(55000,(128,),generator=g);t=torch.randint(64,(128,),generator=g);noise=torch.randn(128,196,generator=g);labels=y[ids2].clone();labels[torch.rand(128,generator=g)<.15]=-1;a=ab[t,None];noisy=a.sqrt()*x[ids2]+(1-a).sqrt()*noise;loss=(net(noisy,t,labels)-noise).square().mean();o.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);o.step();ls.append(loss.item())
  if step%1000==0:hist.append(dict(step=step,training=float(np.mean(ls)),**ev()));ls=[];print(hist[-1],flush=True)
 payload={k:v for k,v in base.items() if k not in ['weights','fixtures','training']};payload['version']=2;payload['weights']={k:v.tolist() for k,v in net.state_dict().items()};payload['training']=dict(seed=731,base_updates=30000,additional_updates=12000,batch=128,learning_rate=.0002,weight_decay=.0001,class_dropout=.15,gradient_clip=1,checkpoint='fixed final update, not selected by validation',before=before,history=hist,validation_seed=1281,validation_probes=2048,sha256=base['training']['sha256'],parameters=base['training']['parameters']);fixtures=[]
 with torch.no_grad():
  for label,t in [(-1,63),(0,31),(3,15),(9,0)]:
   v=torch.linspace(-1,1,196)[None];fixtures.append(dict(input=v[0].tolist(),t=t,label=label,output=net(v,torch.tensor([t]),torch.tensor([label]))[0].tolist()))
 payload['fixtures']=fixtures;Path('1-guided-model.json').write_text(json.dumps(payload,separators=(',',':')))
if __name__=='__main__':main()
