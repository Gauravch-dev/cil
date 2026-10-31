from pathlib import Path
import json
import requests
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE='https://nwdp.nwic.gov.in/api/3/action/'
s=requests.Session();out={}
for name,resource in [('manual_historical','7778f459-0e82-4681-9638-5e494c36fd42'),
                      ('manual_recent','7a5386e8-bf7a-43ac-af78-04565db04e79')]:
    result=s.get(BASE+'datastore_search',params={'resource_id':resource,'fields':'Station,Latitude,Longitude',
        'distinct':'true','filters':json.dumps({'State':'Maharashtra'}),'limit':100},timeout=45).json()
    out[name]={'success':result.get('success'),'total':result.get('result',{}).get('total'),
               'stations':result.get('result',{}).get('records',[])}
    print(name,json.dumps(out[name]),flush=True)
for name,resource in [('rain_historical','f834e1fe-a0e0-4134-a23f-78e3e2eb73b3'),
                      ('rain_recent','1d7a5f00-64a9-484d-a1fa-f77a8ad3f23d')]:
    result=s.get(BASE+'datastore_search',params={'resource_id':resource,'filters':json.dumps({'Station':'KURUNDWAD'}),
        'fields':'Station,Data Acquisition Time,Manual Daily Rainfall (mm)','limit':2000},timeout=45).json()
    assert result['success'];df=pd.DataFrame(result['result']['records'])
    df.to_csv(ROOT/f'kurundwad_{name}.csv',index=False)
    t=pd.to_datetime(df['Data Acquisition Time'],format='%d-%m-%Y %H:%M',errors='coerce')
    v=pd.to_numeric(df['Manual Daily Rainfall (mm)'],errors='coerce')
    out[name]={'api_total':result['result']['total'],'rows_retrieved':len(df),
               'unique_timestamps':int(t.nunique()),'duplicate_timestamps':int(t.duplicated().sum()),
               'zeros':int(v.eq(0).sum()),'negative':int(v.lt(0).sum()),'first_time':str(t.min()),'last_time':str(t.max()),
               'notes':'Sparse observations; absence is not automatically a zero-rainfall day.'}
    print(name,json.dumps(out[name]),flush=True)
(ROOT/'final_source_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

# Match existing published metadata by name, basin and coordinates, not name alone.
metadata=pd.read_csv(ROOT.parents[1]/'experiments/indofloods_severity_v1/raw/metadata_indofloods.csv')
match=metadata.loc[metadata.Station.str.upper().eq('KURUNDWAD')].copy()
match.to_csv(ROOT/'kurundwad_published_threshold_metadata.csv',index=False)
