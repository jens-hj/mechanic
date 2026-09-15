import json,math
from pathlib import Path
root=Path('docs/performance-results/2026-09-14-cpu-scale/ccd-followup')
report={}
for folder in ['cold-diagnostic','expanded-node-cold','packed-check','packed-separated-check','cold','connected','separated']:
 p=root/folder
 if not p.exists():continue
 rows={}
 for file in sorted(p.glob('*.jsonl')):
  records=[json.loads(l) for l in file.read_text().splitlines()]
  ticks=[r for r in records if r.get('kind')=='tick']
  measured=[r for r in ticks if not r['warmup']]
  def percentile(key):
   a=sorted(t[key] for t in measured)
   return a[math.ceil(len(a)*.95)-1] if a else None
  rows[file.stem]={'completed_ticks':len(ticks),'complete':bool(records and records[-1].get('kind')=='summary'), 'degraded_ticks':sum(t['degraded'] for t in ticks), 'p95':{k:percentile(k) for k in ['duration_ms','query_ms','continuous_ms','rows_ms','dynamics_ms','constraints_ms']}, 'mean_ms':sum(t['duration_ms'] for t in measured)/len(measured) if measured else None, 'p99_ms':sorted(t['duration_ms'] for t in measured)[math.ceil(len(measured)*.99)-1] if measured else None, 'maximum_ms':max((t['duration_ms'] for t in measured),default=None), 'compute_tps':len(measured)*1000/sum(t['duration_ms'] for t in measured) if measured else None}
 comparisons={}
 for label,a in rows.items():
  if '-baseline-' not in label:continue
  other=label.replace('-baseline-','-candidate-')
  if other not in rows:continue
  b=rows[other]
  def ticks(n):return [json.loads(l) for l in (p/(n+'.jsonl')).read_text().splitlines() if json.loads(l).get('kind')=='tick']
  ta,tb=ticks(label),ticks(other)
  mismatches=[x['tick'] for x,y in zip(ta,tb) if x['state_hash']!=y['state_hash']]
  counts=['collider_pair_candidates','triangle_candidates','continuous_collider_pair_candidates','continuous_triangle_candidates','continuous_separation_evaluations']
  count_diff=[x['tick'] for x,y in zip(ta,tb) if any(x[k]!=y[k] for k in counts)]
  comparisons[other]={'matching_prefix_hashes':not mismatches,'matching_candidate_counts':not count_diff,'mismatched_ticks':mismatches[:10], 'p95_change_percent':100*(b['p95']['duration_ms']/a['p95']['duration_ms']-1) if a['p95']['duration_ms'] and b['p95']['duration_ms'] else None}
 report[folder]={'runs':rows,'comparisons':comparisons}
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
