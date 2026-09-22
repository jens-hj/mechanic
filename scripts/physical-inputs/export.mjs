// Bake the approved independently authored models into renderer-neutral meshes.
// Run from the repository root: node scripts/physical-inputs/export.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const reference='docs/physical-inputs/concept';
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'mechanic-input-export-'));
try {
fs.copyFileSync(`${reference}/three.module.js`,`${temporary}/three.mjs`);
let source=fs.readFileSync(`${reference}/parts.js`,'utf8').replace("'./three.module.js'","'./three.mjs'");
const start=source.indexOf('const loader=');
const end=source.indexOf('const steel=',start);
source=source.slice(0,start)+`export const textureLoads=[];
function material(name,color){let m=new T.MeshStandardMaterial({color});m.name=name;return m}
`+source.slice(end);
source=source.replace('rest-(pressed?travel:0)','rest-Number(pressed)*travel');
// The reference measured springs from the cap origin. The two broad caps
// extend below that origin; terminate their coils beneath the actual underside.
source=source.replace('span=cap.position.y-strokeBase-s*.018',
 'span=cap.position.y-strokeBase-s*.018-(cm===10?.009:cm===25?.0205:0)');
source=source.replace('end=rest-travel-.006*s', 'end=rest-travel-.006*s-(cm===10?.009:cm===25?.0205:0)');
source += '\nexport const finishes=[steel,silver,rubber,plastic,amber,mint,dark,off,lit];\n';
fs.writeFileSync(`${temporary}/parts.mjs`,source);
const T=await import(pathToFileURL(`${temporary}/three.mjs`));
const {createDial,createButton,finishes}=await import(pathToFileURL(`${temporary}/parts.mjs`));
for(const cm of [5,10,25])for(const kind of ['dial','button']){
 const part=kind==='dial'?createDial(cm):createButton(cm), chunks=[];
 const box=new T.Box3().setFromObject(part.group);
 const height=(box.max.y-box.min.y)*.25, center=(box.max.y+box.min.y)*.125;
 const collect=(springPose=null)=>{
 part.group.updateMatrixWorld(true);
 part.group.traverse(o=>{
  if(!o.isMesh)return;
  const spring=part.springs?.includes(o);
  if(springPose!==null&&!spring)return;
  if(springPose===null&&spring)return;
  let owner=0;
  for(let p=o;p;p=p.parent){if(p===part.rotor)owner=1;if(p===part.cap)owner=2}
  if(spring)owner=3+springPose;
  if(o.userData.guide)owner=6;
  const tick=part.ticks?.indexOf(o)??-1;if(tick>=0)owner=10+tick;
  let finish=finishes.indexOf(o.material);
  if(o.material===part.signal)finish=5;
  if(finish<0)throw Error('unrecognized material');
  const g=o.geometry,positions=[],normals=[],uvs=[], normalMatrix=new T.Matrix3().getNormalMatrix(o.matrixWorld);
  for(let i=0;i<g.attributes.position.count;i++){
   const p=new T.Vector3().fromBufferAttribute(g.attributes.position,i).applyMatrix4(o.matrixWorld).multiplyScalar(.25);p.y-=center;
   positions.push(...p.toArray());
   normals.push(...new T.Vector3().fromBufferAttribute(g.attributes.normal,i).applyNormalMatrix(normalMatrix).toArray());
   uvs.push(g.attributes.uv.getX(i),g.attributes.uv.getY(i));
  }
  const indices=g.index?Array.from(g.index.array):Array.from({length:g.attributes.position.count},(_,i)=>i);
  chunks.push({owner,finish,positions,normals,uvs,indices});
 });
 };
 collect();
 if(kind==='button')for(const [pose,value] of [0,.5,1].entries()){part.update(value);collect(pose)}
 const buffers=[];
 const uint=v=>{const b=Buffer.alloc(4);b.writeUInt32LE(v);buffers.push(b)};
 const float=v=>{const b=Buffer.alloc(4);b.writeFloatLE(v);buffers.push(b)};
 uint(chunks.length);
 for(const c of chunks){uint(c.owner);uint(c.finish);uint(c.positions.length/3);uint(c.indices.length);for(const list of [c.positions,c.normals,c.uvs])for(const v of list)float(v);for(const i of c.indices)uint(i)}
 fs.writeFileSync(`crates/mechanic-core/assets/physical-inputs/${kind}-${cm}.bin`,Buffer.concat(buffers));
 console.log(`${kind} ${cm}: height ${height.toFixed(9)}, ${chunks.length} chunks`);
}
} finally {fs.rmSync(temporary,{recursive:true,force:true})}
