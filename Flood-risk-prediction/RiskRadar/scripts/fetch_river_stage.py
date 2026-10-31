"""Download only three approved CWC gauges, preserving raw official records."""
from pathlib import Path
import json
from hashlib import sha256
from datetime import datetime, timezone
import requests
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/river_stage/raw'
RESOURCES={'historical':'7778f459-0e82-4681-9638-5e494c36fd42',
           'recent':'7a5386e8-bf7a-43ac-af78-04565db04e79'}
CANDIDATES={'kurundwad':['KURUNDWAD'], 'narsingpur':['Narsingpur','NARSINGPUR'], 'takli':['Takli','TAKLI']}
API='https://nwdp.nwic.gov.in/api/3/action/datastore_search'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_metadata=pd.read_csv(ROOT/'experiments/indofloods_severity_v1/raw/metadata_indofloods.csv')
    manifest={'source':'CWC / National Water Data Portal, NWIC', 'endpoint':API,
              'retrieved_utc':datetime.now(timezone.utc).isoformat(),'resources':[]}
    session=requests.Session()
    for key,names in CANDIDATES.items():
        meta=source_metadata.loc[source_metadata.Station.str.upper().eq(key.upper())].iloc[0]
        for period,resource in RESOURCES.items():
            path=OUT/f'{key}_{period}.csv'
            if path.exists():
                df=pd.read_csv(path)
                chosen=df.Station.iloc[0]
                total=len(df)
            else:
                chosen=None
                for name in names:
                    params={'resource_id':resource,'filters':json.dumps({'Station':name,'State':'Maharashtra'}),'limit':1}
                    response=session.get(API,params=params,timeout=(15,45));response.raise_for_status()
                    payload=response.json();assert payload.get('success'),payload
                    total=payload['result']['total']
                    if total:
                        chosen=name;break
                if chosen is None:
                    raise RuntimeError(f'No official Maharashtra records found for {key}/{period}')
                rows=[]
                for offset in range(0,total,5000):
                    response=session.get(API,params={'resource_id':resource,
                        'filters':json.dumps({'Station':chosen,'State':'Maharashtra'}),
                        'limit':5000,'offset':offset},timeout=(15,60));response.raise_for_status()
                    payload=response.json();assert payload['success']
                    rows.extend(payload['result']['records'])
                assert len(rows)==total
                df=pd.DataFrame(rows)
                assert df.State.eq('Maharashtra').all()
                assert (pd.to_numeric(df.Latitude)-meta.Latitude).abs().max()<.03
                assert (pd.to_numeric(df.Longitude)-meta.Longitude).abs().max()<.03
                df.to_csv(path,index=False)
            manifest['resources'].append({'station':key,'name':chosen,'period':period,'resource_id':resource,
                'rows':total,'file':path.name,'sha256':sha256(path.read_bytes()).hexdigest(),
                'latitude':float(pd.to_numeric(df.Latitude).median()),
                'longitude':float(pd.to_numeric(df.Longitude).median()),
                'zero_gauge_rl_values':pd.to_numeric(df.RL_of_zeroGauge,errors='coerce').dropna().unique().tolist()})
            print(key,period,chosen,total,'records',flush=True)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
