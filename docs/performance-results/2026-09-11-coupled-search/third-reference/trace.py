#!/usr/bin/env python3
"""Frozen central-path predictor/corrector control; output is not a solution.

The runtime's 128-row response limit does not apply to this offline control.
The smoothed contact equations only generate candidates. Rust regressions then
check original normal, tangent and rolling laws independently of this script.
"""
import argparse
import numpy as np
import ast, hashlib, json, math, sys
from pathlib import Path

script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
def read(node):
    if isinstance(node, (ast.List, ast.Tuple)): return [read(x) for x in node.elts]
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -read(node.operand)
    if isinstance(node, ast.Name): return {'inf': math.inf, 'true': True, 'false': False}[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Some' and len(node.args) == 1: return read(node.args[0])
    raise ValueError(ast.dump(node))


arguments=argparse.ArgumentParser()
arguments.add_argument('--fixture',type=Path,required=True)
arguments.add_argument('--output',type=Path,required=True)
args=arguments.parse_args()
mass,blocks=read(ast.parse(args.fixture.read_text(),mode='eval').body)
for b in blocks:
    rows=sum(5 if r is not None else 3 for mus,muk,sliding,r in b[3])
    if not b[3] or rows!=len(b[0]) or rows!=len(b[1]) or rows!=len(b[2]):
        raise ValueError('reference requires complete contact-only blocks')
    cursor=0
    for mus,muk,sliding,r in b[3]:
        count=5 if r is not None else 3
        if not all(math.isfinite(c) and c>=0 for c in [mus,muk]+([r] if r is not None else [])):
            raise ValueError('invalid friction coefficients')
        if b[2][cursor]!=[0,math.inf] or any(bounds!=[-math.inf,math.inf] for bounds in b[2][cursor+1:cursor+count]):
            raise ValueError('reference requires unbounded unilateral contact laws')
        cursor+=count
J=np.array([r for b in blocks for r in b[0]]); n=J.shape[1]; m=len(J)
H=np.array(mass).reshape(n,n)
if not np.all(np.isfinite(H)) or not np.all(np.isfinite(J)) or not np.allclose(H,H.T,rtol=0,atol=1e-12):
    raise ValueError('invalid dynamics')
np.linalg.cholesky(H)
W=J@np.linalg.solve(H,J.T)
y=np.array([t for b in blocks for t in b[1]])
if not np.all(np.isfinite(y)):raise ValueError('invalid targets')
scale=np.zeros(m); points=[]; k=0
for b in blocks:
    size=len(b[1]); scale[k:k+size]=np.max(np.sum(np.abs(W[k:k+size,k:k+size]),axis=1))
    for mus,muk,sliding,rolling in b[3]:
        points.append((k,muk if sliding else mus,rolling)); k+=5 if rolling is not None else 3

def evaluate(x):
    at=x+(y-W@x)/scale; p=at.copy(); P=np.eye(m)
    for k,mu,r in points:
        p[k]=max(at[k],0); P[k,k]=float(at[k]>0)
        for offset,c in [(1,mu)]+([(3,r)] if r is not None else []):
            sl=slice(k+offset,k+offset+2); v=at[sl]; length=np.linalg.norm(v); radius=c*p[k]
            if length>radius:
                u=v/length; p[sl]=u*radius
                P[sl,sl]=radius/length*(np.eye(2)-np.outer(u,u)); P[sl,k]=c*u*P[k,k]
    F=(x-p)*scale
    A=(np.eye(m)-P@(np.eye(m)-W/scale[:,None]))*scale[:,None]
    return F,A
x=np.zeros(m)
original=evaluate

def smooth(x,epsilon):
    at=x+(y-W@x)/scale; p=at.copy(); P=np.zeros((m,m))
    for k,mu,r in points:
        e=epsilon/scale[k]; a=at[k]; root=np.hypot(a,e)
        # Preserve the small positive root and its derivative when a < 0.
        # Direct subtraction cancels both at late smoothing stages.
        p[k]=0.5*(a+root) if a>=0 else 0.5*e*(e/(root-a))
        P[k,k]=p[k]/root
        for offset,c in [(1,mu)]+([(3,r)] if r is not None else []):
            sl=slice(k+offset,k+offset+2); v=at[sl]; l=np.linalg.norm(v); radius=c*p[k]
            root=np.hypot(l-radius,e); den=0.5*(radius+l+root); dl=0.5*(1+(l-radius)/root); dr=1-dl
            p[sl]=v*radius/den
            P[sl,sl]=radius/den*np.eye(2)-radius/den**2*dl*np.outer(v,v)/max(l,1e-300)
            P[sl,k]=v*c*(1/den-radius/den**2*dr)*P[k,k]
    return (x-p)*scale,(np.eye(m)-P@(np.eye(m)-W/scale[:,None]))*scale[:,None]


def central(x,epsilon):
    F=W@x-y;A=W.copy()
    for k,mu,r in points:
        normal=x[k]
        if normal<=0:return None,None
        F[k]-=epsilon/normal;A[k,k]+=epsilon/normal**2
        for offset,c in [(1,mu)]+([(3,r)] if r is not None else []):
            sl=slice(k+offset,k+offset+2);t=x[sl];D=(c*normal-np.linalg.norm(t))*(c*normal+np.linalg.norm(t))
            if D<=0:return None,None
            F[sl]+=2*epsilon*t/D
            A[sl,sl]+=2*epsilon/D*np.eye(2)+4*epsilon/D**2*np.outer(t,t)
            A[sl,k]-=4*epsilon*c*c*normal*t/D**2
    return F,A
x=np.zeros(m)
for k,mu,r in points:x[k]=10
for eps in [10.**(-k) for k in range(0,6)]:
    for iteration in range(128):
        F,A=central(x,eps);norm=np.linalg.norm(F)
        if max(abs(F))<max(1e-11,eps*1e-5):break
        row=np.max(abs(A),axis=1);d=np.linalg.lstsq(A/row[:,None],-F/row,rcond=1e-14)[0];best=(norm,x)
        for power in range(40):
            trial=x+0.5**power*d;f,_=central(trial,eps)
            if f is None:continue
            merit=np.linalg.norm(f)
            if merit<best[0]:best=merit,trial
            if merit<norm*(1-1e-4*0.5**power):break
        if best[0]>=norm:break
        x=best[1]
    err=max(abs(original(x)[0]));print('central',eps,iteration,'equation',max(abs(central(x,eps)[0])),'original',err,flush=True)
    args.output.write_text(repr(x.tolist()))
    if err<1e-10:break

unit=np.ones(m)
for k,mu,r in points:unit[k:k+3]=100;unit[k+3:k+5]=100*r
z=np.r_[x/unit,np.log(eps)];previous=np.zeros(m+1);previous[-1]=-1;h=.25
for step in range(200):
    x=z[:-1]*unit;eps=np.exp(z[-1]);F,A=central(x,eps)
    derivative=np.zeros(m)
    for k,mu,r in points:
        derivative[k]=-eps/x[k]
        for o,c in [(1,mu),(3,r)]:
            sl=slice(k+o,k+o+2);t=x[sl];D=(c*x[k]-np.linalg.norm(t))*(c*x[k]+np.linalg.norm(t));derivative[sl]=2*eps*t/D
    tangent_matrix=np.column_stack((A*unit,derivative));rs=np.max(abs(tangent_matrix),axis=1)
    _,singular,vt=np.linalg.svd(tangent_matrix/rs[:,None],full_matrices=True);tangent=vt[-1]
    if tangent@previous<0:tangent=-tangent
    success=False
    for retry in range(12):
        predicted=z+h*tangent;trial=predicted.copy()
        for iteration in range(24):
            ex=np.exp(trial[-1]);tx=trial[:-1]*unit;f,a=central(tx,ex)
            if f is None:break
            arc=tangent@(trial-predicted);res=np.r_[f,arc]
            if max(abs(f))<1e-10 and abs(arc)<1e-10:success=True;break
            de=np.zeros(m)
            for k,mu,r in points:
                de[k]=-ex/tx[k]
                for o,c in [(1,mu),(3,r)]:
                    sl=slice(k+o,k+o+2);t=tx[sl];D=(c*tx[k]-np.linalg.norm(t))*(c*tx[k]+np.linalg.norm(t));de[sl]=2*ex*t/D
            jac=np.vstack((np.column_stack((a*unit,de)),tangent));rs2=np.max(abs(jac),axis=1);delta=np.linalg.lstsq(jac/rs2[:,None],-res/rs2,rcond=1e-14)[0]
            norm=np.linalg.norm(res/rs2);found=False
            for power in range(32):
                test=trial+delta*.5**power
                if abs(test[-1])>100:continue
                tf,_=central(test[:-1]*unit,np.exp(test[-1]))
                if tf is None:continue
                merit=np.linalg.norm(np.r_[tf,tangent@(test-predicted)]/rs2)
                if merit<norm:trial=test;found=True;break
            if not found:break
        if success:break
        h*=.5
    if not success:print('arc failed',step,h,flush=True);break
    z=trial;previous=tangent;x=z[:-1]*unit;eps=np.exp(z[-1]);err=max(abs(original(x)[0]))
    print('arc',step,'epsilon',eps,'step',h,'iterations',iteration,'original',err,flush=True);args.output.write_text(repr(x.tolist()))
    if err<1e-10:break
    if iteration<5:h=min(.5,h*1.5)
