'use strict';
const GrainBase=typeof module!=='undefined'?require('./engine.js'):null;
function noisePrediction(x,t,label,m){return GrainBase?GrainBase.denoise(x,t,label,m):denoise(x,t,label,m)}
function noiseRng(seed){return GrainBase?new GrainBase.RNG(seed):new RNG(seed)}
function guidedNoise(x,t,label,m,scale=1){if(!Number.isFinite(scale)||scale<0||scale>5)throw Error('Guidance must be 0–5');let c=noisePrediction(x,t,label,m);if(scale===1)return c;let u=noisePrediction(x,t,-1,m);return u.map((v,i)=>v+scale*(c[i]-v))}
function timeGrid(count){if(![8,16,32,64].includes(count))throw Error('Choose 8, 16, 32 or 64 steps');return Array.from({length:count},(_,i)=>Math.round(63*(1-i/(count-1))))}
function ddim(x,t,next,label,m,scale=1){let eps=guidedNoise(x,t,label,m,scale),a=m.alpha_bar[t],b=next<0?1:m.alpha_bar[next],clean=x.map((v,i)=>Math.max(-1,Math.min(1,(v-Math.sqrt(1-a)*eps[i])/Math.sqrt(a))));let corrected=x.map((v,i)=>(v-Math.sqrt(a)*clean[i])/Math.sqrt(1-a));return{pixels:clean.map((v,i)=>Math.sqrt(b)*v+Math.sqrt(1-b)*corrected[i]),clean,epsilon:eps}}
function initial(seed,seedB,mix){if(!Number.isInteger(seed)||seed<0||seed>4294967295||!Number.isInteger(seedB)||seedB<0||seedB>4294967295||!Number.isFinite(mix)||mix<0||mix>1)throw Error('Invalid seed or noise blend');let a=noiseRng(seed),b=noiseRng(seedB),angle=mix*Math.PI/2;return Array.from({length:196},()=>Math.cos(angle)*a.normal()+Math.sin(angle)*b.normal())}
function sample(seed,label,m,count=16,scale=1,seedB=43,mix=0){if(!Number.isInteger(label)||label<0||label>9)throw Error('Invalid digit');let grid=timeGrid(count),x=initial(seed,seedB,mix),frames=[x];for(let i=0;i<grid.length;i++){x=ddim(x,grid[i],grid[i+1]??-1,label,m,scale).pixels;frames.push(x)}return frames}
if(typeof module!=='undefined'){module.exports={guidedNoise,timeGrid,ddim,initial,sample}}
