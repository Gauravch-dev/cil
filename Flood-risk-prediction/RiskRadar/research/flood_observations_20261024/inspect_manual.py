from pathlib import Path
import json
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = 'https://nwdp.nwic.gov.in/api/3/action/'
session = requests.Session()
payload = session.get(BASE+'package_search', params={'q':'title:river AND title:water AND title:level','rows':20},timeout=40).json()
(ROOT/'river_level_catalog.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
matches=[]
for package in payload.get('result',{}).get('results',[]):
    print('PACKAGE',package.get('name'),flush=True)
    for r in package.get('resources',[]):
        name=r.get('name','')
        if 'krishna' in name.lower() and ('manual' in name.lower()) and ('1991' in name or '2021' in name):
            item={'package':package['name'],**{k:r.get(k) for k in ['id','name','url','datastore_active','size']}}
            matches.append(item); print('RESOURCE',json.dumps(item),flush=True)
            for case in ['KURUNDWAD','Kurundwad']:
                query={'resource_id':r['id'],'filters':json.dumps({'Station':case}),'limit':5}
                response=session.get(BASE+'datastore_search',params=query,timeout=40).json()
                (ROOT/f"manual_sample_{r['id']}_{case}.json").write_text(json.dumps(response,indent=2),encoding='utf-8')
                result=response.get('result',{})
                print('STATION_SAMPLE',json.dumps({'resource':r['id'],'case':case,'success':response.get('success'),
                      'total':result.get('total'),'records':result.get('records',[])[:2]}),flush=True)
(ROOT/'manual_resource_matches.json').write_text(json.dumps(matches,indent=2),encoding='utf-8')

for period in ['historical','recent']:
    path=ROOT/f'kurundwad_{period}_observations.csv'
    df=pd.read_csv(path);col='River Water Level Telemetry Hourly (meter)';levels=pd.to_numeric(df[col],errors='coerce')
    summary={'period':period,'raw_at_or_above537_16':int(levels.ge(537.16).sum()),
             'values_between537_16_and550':int(levels.between(537.16,550).sum()),
             'values_above550':int(levels.gt(550).sum()),'most_common_levels':levels.value_counts().head(5).to_dict()}
    print('TELEMETRY_QUALITY',json.dumps(summary),flush=True)
