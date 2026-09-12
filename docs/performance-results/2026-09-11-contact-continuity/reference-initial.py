import numpy as np
from pathlib import Path
exec(Path('docs/performance-results/2026-09-10-cold-settling/contact-mode-probe.py').read_text().split('def run(')[0])
W=np.array(w); target=np.array([t for b in blocks for t in b[1]])
scale=np.array([s for _,size,s in layout for _ in range(size)])
print('J singular',np.linalg.svd(np.array(rows),compute_uv=False))
contacts=[i for b,(i,_,_) in zip(blocks,layout) if b[3]]
for indices in [[i for c in contacts for i in (c,c+1,c+2)], [c for c in contacts]+[i for c in contacts[:2] for i in (c+1,c+2)]]:
    A=np.array(rows)[indices]; y=target[indices]; x,_,rank,s=np.linalg.lstsq(A,y,rcond=None)
    print('sticking targets',indices,'rank',rank,'singular',s,'error',max(abs(A@x-y)))

def evaluate(x, derivative=False):
    at=x+(target-W@x)/scale
    P=np.zeros_like(W); projected=at.copy()
    for b,(first,size,_) in zip(blocks,layout):
        for i,(lo,hi) in enumerate(b[2]):
            j=first+i; projected[j]=np.clip(at[j],lo,hi); P[j,j]=float(lo<at[j]<hi)
        k=first
        for mus,muk,sliding,rolling in b[3]:
            for offset,mu in [(1,muk if sliding else mus)]+([(3,rolling)] if rolling is not None else []):
                sl=slice(k+offset,k+offset+2); v=projected[sl].copy(); length=np.linalg.norm(v); radius=mu*projected[k]
                if length>radius:
                    u=v/length; projected[sl]=u*radius
                    P[sl,sl]=radius/length*(np.eye(2)-np.outer(u,u)); P[sl,k]=mu*u*P[k,k]
            k+=5 if rolling is not None else 3
    F=(x-projected)*scale
    return F, (np.eye(len(x))-P@(np.eye(len(x))-W/scale[:,None]))*scale[:,None]

x=np.array(warm)
for iteration in range(500):
    F,A=evaluate(x); norm=np.linalg.norm(F)
    if iteration%25==0 or max(abs(F))<1e-10: print('iter',iteration,'res',max(abs(F)), 'norm',norm,flush=True)
    if max(abs(F))<1e-10: break
    best=(norm,x)
    for damping in [0,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2]:
        if damping==0: delta=np.linalg.lstsq(A,-F,rcond=1e-14)[0]
        else: delta=np.linalg.solve(A.T@A+damping*np.eye(len(x)),-A.T@F)
        for power in range(20):
            trial=x+delta*0.5**power; f,_=evaluate(trial); merit=np.linalg.norm(f)
            if merit<best[0]: best=(merit,trial)
            if merit<norm*(1-1e-4*0.5**power): break
    if best[0]>=norm: print('stalled'); break
    x=best[1]
print('impulses',x.tolist())
print('slack',(W@x-target).tolist())
for c in contacts: print('contact',c,'fraction',np.linalg.norm(x[c+1:c+3])/(blocks[6+(c-6)//5][3][0][0 if c==6 else 1]*x[c]))
np.savez('/private/tmp/mechanic-contact-reference.npz',impulses=x,W=W,targets=target)
