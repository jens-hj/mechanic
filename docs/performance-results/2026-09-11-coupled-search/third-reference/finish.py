#!/usr/bin/env python3
"""Independent dense NumPy search reference; no physical tick or speed acceptance.

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
arguments.add_argument('--initial',type=Path,required=True)
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


def eliminated(x,eps):
    v=W@x-y;F=v.copy();A=W.copy()
    for k,mu,r in points:
        normal=x[k]
        if normal<=0:return None,None
        F[k]-=eps/normal;A[k,k]+=eps/normal**2
        for o,c in [(1,mu)]+([(3,r)] if r is not None else []):
            sl=slice(k+o,k+o+2);u=v[sl];t=x[sl];R=c*normal;square=u@u;root=np.hypot(eps,R*np.sqrt(square));den=eps+root;beta=R*R/den
            derivative=-(R**4/(root*den**2))*u@W[sl]
            derivative[k]+=c*(2*R/den-R**3*square/(root*den**2))
            divisor=1+scale[k]*beta
            F[sl]=scale[k]*(t+beta*u)/divisor
            A[sl]=scale[k]/divisor*(np.eye(m)[sl]+beta*W[sl])+np.outer(scale[k]*(u-scale[k]*t)/divisor**2,derivative)
    return F,A
x=np.array(ast.literal_eval(args.initial.read_text()))
for eps in [1.31e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20]:
    for iteration in range(128):
        err=max(abs(original(x)[0]))
        if err<1e-9:break
        F,A=eliminated(x,eps);norm=np.linalg.norm(F)
        if max(abs(F))<1e-11:break
        row=np.max(abs(A),axis=1);d=np.linalg.lstsq(A/row[:,None],-F/row,rcond=1e-14)[0];best=(norm,x)
        for power in range(40):
            trial=x+d*.5**power;f,_=eliminated(trial,eps)
            if f is None:continue
            merit=np.linalg.norm(f)
            if merit<best[0]:best=merit,trial
            if merit<norm*(1-1e-4*.5**power):break
        if best[0]>=norm:break
        x=best[1]
    err=max(abs(original(x)[0]));print('eliminated',eps,iteration,'equation',max(abs(eliminated(x,eps)[0])),'original',err,flush=True);args.output.write_text(repr(x.tolist()))
    if err<1e-9:break

# Validate the original laws, never the central-path residual.

at=x+(y-W@x)/scale
for k,mu,r in points:
    at[k]=max(0,at[k])
    for offset,c in [(1,mu)]+([(3,r)] if r is not None else []):
        sl=slice(k+offset,k+offset+2); length=np.linalg.norm(at[sl])
        if length>c*at[k]:at[sl]*=c*at[k]/length
x=at
velocity=np.linalg.solve(H,J.T@x)
residual=float(max(abs(original(x)[0])))
normal_error=0.0; disk_error=0.0; direction_error=0.0
for k,mu,r in points:
    normal=x[k]; gap=J[k]@velocity-y[k]
    normal_error=max(normal_error,-gap,abs(gap) if normal>1e-8 else 0)
    for offset,c in [(1,mu)]+([(3,r)] if r is not None else []):
        sl=slice(k+offset,k+offset+2); impulse=x[sl]; slip=J[sl]@velocity-y[sl]
        radius=c*normal; load=np.linalg.norm(impulse); speed=np.linalg.norm(slip)
        disk_error=max(disk_error,load-radius)
        if speed>1e-8:
            disk_error=max(disk_error,abs(load-radius))
            direction_error=max(direction_error,float(impulse@slip),
                abs(impulse[0]*slip[1]-impulse[1]*slip[0])/max(radius,1e-12))
converged=bool(np.all(np.isfinite(x)) and np.all(np.isfinite(velocity))
    and residual<=1e-9 and normal_error<=1e-9
    and disk_error<=1e-8 and direction_error<=1e-8)
status={"converged":converged,"original_residual":residual,
    "normal_error":float(normal_error),"disk_error":float(disk_error),
    "direction_error":float(direction_error),"rows":m,"coordinates":n,
    "numpy_version":np.__version__,"fixture_sha256":hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
    "script_sha256":script_sha256,
    "semantics":"instantaneous algebra candidate; no physical tick or performance acceptance"}
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=script_sha256:
    raise RuntimeError('reference source changed during the run')
status["initial_sha256"]=hashlib.sha256(args.initial.read_bytes()).hexdigest()
args.output.write_text(repr(x.tolist()))
args.output.with_suffix('.velocity.ron').write_text(repr(velocity.tolist()))
args.output.with_suffix('.status.json').write_text(json.dumps(status,indent=2)+"\n")
print(json.dumps(status),flush=True)
sys.exit(0 if converged else 1)
