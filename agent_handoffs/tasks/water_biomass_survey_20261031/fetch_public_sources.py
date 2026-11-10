"""Read-only public source retrieval; write only new survey evidence."""
from pathlib import Path
import datetime,hashlib,json,urllib.request
p=Path(__file__).resolve().parent;out=p/'public_sources';out.mkdir(exist_ok=False)
base='https://raw.githubusercontent.com/DrylandEcology/STEPWAT2/003cb9b4adbf13d64a39d6d62fd95280adac27fc/'
items=[(n,base+n) for n in ['ST_resgroups.c','ST_species.c','ST_indivs.c']]
items += [(n+'.html','https://www.waveslab.org/publications/'+n) for n in ['kelly-caylor20041479','kelly-caylor20051459','kelly-caylor20041398']]
records=[]
for name,url in items:
 rec={'name':name,'url':url,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'LEAF-public-literature-survey/1.0'})
  with urllib.request.urlopen(req,timeout=15) as response:
   data=response.read(5000001);assert len(data)<=5000000
   rec.update({'status':response.status,'final_url':response.url,'content_type':response.headers.get('Content-Type'),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
   (out/name).write_bytes(data)
 except Exception as e:rec.update({'error':type(e).__name__+': '+str(e)})
 records.append(rec);print(json.dumps(rec),flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
