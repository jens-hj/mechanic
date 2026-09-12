import ast, json, math
from pathlib import Path

def read(node):
    if isinstance(node, (ast.List, ast.Tuple)): return [read(x) for x in node.elts]
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -read(node.operand)
    if isinstance(node, ast.Name): return {'inf': math.inf, 'true': True, 'false': False}[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Some' and len(node.args) == 1: return read(node.args[0])
    raise ValueError(ast.dump(node))

source = Path('docs/performance-results/2026-09-10-cold-settling/unconverged-cold-settling.ron')
lower, blocks, warm = read(ast.parse(source.read_text(), mode='eval').body)
rows = [row for b in blocks for row in b[0]]
n = len(rows[0])
def inverse(rhs):
    v = list(rhs)
    for i in range(n): v[i] = (v[i] - sum(lower[i*n+j]*v[j] for j in range(i))) / lower[i*n+i]
    for i in reversed(range(n)): v[i] = (v[i] - sum(lower[j*n+i]*v[j] for j in range(i+1,n))) / lower[i*n+i]
    return v
columns = [inverse(row) for row in rows]
w = [[sum(a*b for a,b in zip(row,col)) for col in columns] for row in rows]
layout=[]; first=0
for block in blocks:
    size=len(block[1]); scale=max(sum(abs(w[i][j]) for j in range(first,first+size)) for i in range(first,first+size))
    layout.append((first,size,scale)); first+=size

def project(block, values, kinetic):
    result=[min(max(x,lo),hi) for x,(lo,hi) in zip(values,block[2])]
    row=0
    for mu_s,mu_k,sliding,rolling in block[3]:
        mu = mu_k if sliding or kinetic else mu_s
        for offset,radius in [(1,mu*result[row])]+([(3,rolling*result[row])] if rolling is not None else []):
            a,b=result[row+offset:row+offset+2]; length=math.hypot(a,b)
            if length>radius:
                result[row+offset]=a*radius/length; result[row+offset+1]=b*radius/length
        row+=5 if rolling is not None else 3
    return result

def run(kinetic):
    impulse=list(warm) if warm is not None else [0.0]*len(rows)
    for block,(first,size,_) in zip(blocks,layout): impulse[first:first+size]=project(block,impulse[first:first+size],kinetic)
    history=[]
    for iteration in range(1,8193):
        for block,(first,size,scale) in zip(blocks,layout):
            current=[sum(a*b for a,b in zip(w[first+i],impulse)) for i in range(size)]
            candidate=[impulse[first+i]+(block[1][i]-current[i])/scale for i in range(size)]
            impulse[first:first+size]=project(block,candidate,kinetic)
        if iteration in [256,8192]:
            response=[sum(a*b for a,b in zip(row,impulse)) for row in w]
            residual=0.0
            for block,(first,size,scale) in zip(blocks,layout):
                candidate=project(block,[impulse[first+i]+(block[1][i]-response[first+i])/scale for i in range(size)],kinetic)
                residual=max(residual,max(abs(a-impulse[first+i])*scale for i,a in enumerate(candidate)))
            history.append({'iterations':iteration,'projected_velocity_residual':residual})
    contacts=[]
    for block,(first,size,scale) in zip(blocks,layout):
        row=0
        for mu_s,mu_k,sliding,rolling in block[3]:
            mu=mu_k if sliding or kinetic else mu_s
            normal=impulse[first+row]
            contacts.append({'initially_sliding':sliding,'normal_impulse':normal,'normal_slack':response[first+row]-block[1][row], 'tangent_velocity':math.hypot(response[first+row+1]-block[1][row+1],response[first+row+2]-block[1][row+2]),'tangent_cone_fraction':math.hypot(impulse[first+row+1],impulse[first+row+2])/(mu*normal) if normal>0 else None})
            row+=5 if rolling is not None else 3
    return {'forced_kinetic_control':kinetic,'history':history,'contacts':contacts}
print(json.dumps({'kind':'independent_python_pgs_control_not_physical_acceptance','generalized':n,'rows':len(rows),'results':[run(False),run(True)]},indent=2))
