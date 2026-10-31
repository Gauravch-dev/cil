"""Retrieve one matching gauge's manual observations, audit raw feasibility only."""
from pathlib import Path
import json
import requests
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE='https://nwdp.nwic.gov.in/api/3/action/'
session=requests.Session()
resources={'historical':'7778f459-0e82-4681-9638-5e494c36fd42','recent':'7a5386e8-bf7a-43ac-af78-04565db04e79'}
summary={}
for period,resource in resources.items():
    params={'resource_id':resource,'filters':json.dumps({'Station':'KURUNDWAD'}),'limit':1}
    payload=session.get(BASE+'datastore_search',params=params,timeout=45).json()
    assert payload['success']
    total=payload['result']['total']; rows=[]
    cols=['Station','Data Acquisition Time','River Water Level Manual Hourly (meter)','RL_of_zeroGauge','Latitude','Longitude']
    for offset in range(0,total,5000):
        params.update(fields=','.join(cols),limit=5000,offset=offset)
        response=session.get(BASE+'datastore_search',params=params,timeout=45)
        response.raise_for_status(); block=response.json();assert block['success']
        rows.extend(block['result']['records'])
    df=pd.DataFrame(rows);assert len(df)==total
    df.to_csv(ROOT/f'kurundwad_manual_{period}.csv',index=False)
    t=pd.to_datetime(df['Data Acquisition Time'],format='%d-%m-%Y %H:%M',errors='coerce')
    v=pd.to_numeric(df['River Water Level Manual Hourly (meter)'],errors='coerce')
    # Diagnostic envelope only, not a cleaned label definition or automatic repair.
    plausible=v.between(519.455,551.35)
    info={'resource_id':resource,'rows':len(df),'first_time':str(t.min()),'last_time':str(t.max()),
          'invalid_times':int(t.isna().sum()),'invalid_numeric_values':int(v.isna().sum()),
          'duplicate_timestamps':int(t.duplicated().sum()),'raw_min':float(v.min()),'raw_max':float(v.max()),
          'diagnostic_envelope_outside':int((~plausible).sum()),
          'raw_danger_exceedance_readings':int(v.ge(537.16).sum()),
          'diagnostic_plausible_danger_readings':int((plausible&v.ge(537.16)).sum()),
          'plausible_above_examples':df.loc[plausible&v.ge(537.16),cols].head(5).to_dict('records'),
          'caution':'Provisional diagnostics, not cleaned labels. Check dates, datum, flat lines, gaps and official threshold history.'}
    valid=pd.DataFrame({'t':t,'level':v}).loc[plausible&t.notna()].sort_values('t')
    byyear=valid.assign(year=valid.t.dt.year).groupby('year').agg(readings=('level','size'),max_level=('level','max'))
    info['yearly_diagnostics']=byyear.reset_index().to_dict('records')
    summary[period]=info;print(period,json.dumps(info),flush=True)
for period,resource in [('historical','f834e1fe-a0e0-4134-a23f-78e3e2eb73b3'),('recent','1d7a5f00-64a9-484d-a1fa-f77a8ad3f23d')]:
    payload=session.get(BASE+'datastore_search',params={'resource_id':resource,
        'filters':json.dumps({'Station':'KURUNDWAD'}),'limit':3},timeout=45).json()
    (ROOT/f'kurundwad_rainfall_{period}_sample.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    item={'success':payload.get('success'),'total':payload.get('result',{}).get('total'),
          'records':payload.get('result',{}).get('records',[])[:2]}
    summary[f'rainfall_{period}']=item;print('RAINFALL',period,json.dumps(item),flush=True)
(ROOT/'manual_feasibility_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
