"""Independent NumPy float64 DDIM/CFG trajectories, exported weight evaluation."""
import json,math
from pathlib import Path
import numpy as np
class RNG:
 def __init__(self,seed):self.v=seed
 def u(self):self.v=(1664525*self.v+1013904223)%2**32;return(self.v+.5)/2**32
 def n(self):return math.sqrt(-2*math.log(self.u()))*math.cos(2*math.pi*self.u())
def denoise(x,t,label,m):
 w={k:np.array(v) for k,v in m['weights'].items()};a=t*np.exp(-math.log(10000)*np.arange(8)/7);c=np.zeros(10)
 if label>=0:c[label]=1
 h=np.concatenate([x,np.sin(a),np.cos(a),c]);h=w['fc1.weight']@h+w['fc1.bias'];h=h/(1+np.exp(-h));h=w['fc2.weight']@h+w['fc2.bias'];h=h/(1+np.exp(-h));return w['fc3.weight']@h+w['fc3.bias']
def trajectory(seed,seedb,mix,label,count,scale,m):
 r=RNG(seed);s=RNG(seedb);x=np.array([math.cos(mix*math.pi/2)*r.n()+math.sin(mix*math.pi/2)*s.n() for _ in range(196)]);frames=[x.tolist()];grid=np.rint(np.linspace(63,0,count)).astype(int)
 # The tested 8/16/32/64 grids have no .5 tie points, so NumPy/JS round agree.
 for i,t in enumerate(grid):
  eps=denoise(x,t,label,m)
  if scale!=1:u=denoise(x,t,-1,m);eps=u+scale*(eps-u)
  a=m['alpha_bar'][t];b=m['alpha_bar'][grid[i+1]] if i+1<count else 1;clean=np.clip((x-np.sqrt(1-a)*eps)/np.sqrt(a),-1,1);corrected=(x-np.sqrt(a)*clean)/np.sqrt(1-a);x=np.sqrt(b)*clean+np.sqrt(1-b)*corrected;frames.append(x.tolist())
 return frames
if __name__=='__main__':
 m=json.loads(Path('1-guided-model.json').read_text());out=[]
 for seed,seedb,mix,label,count,scale in [(42,917,0,3,16,3),(123,917,.5,7,8,0),(42,917,1,0,32,1),(5,8,.25,9,64,5)]:out.append(dict(seed=seed,seedB=seedb,mix=mix,label=label,count=count,scale=scale,frames=trajectory(seed,seedb,mix,label,count,scale,m)))
 Path('2-sampling-fixtures.json').write_text(json.dumps(out,separators=(',',':')))
