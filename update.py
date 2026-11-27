import json,re,time,datetime,pathlib
import requests,numpy as np
from bs4 import BeautifulSoup
from scipy.optimize import minimize
from scipy.stats import poisson,skellam
from zoneinfo import ZoneInfo
TEAMS=['FUTSAL TEVERE','ROMA UNO','REAL MATTEI','GIOVANNI CASTELLO','DISTRICT SEVEN','OLIMPUS TIBURTINA ROMA','IL CASALE DI ROMA','ITALIAN KICK OFF','POLISPORTIVA V','PARROCCHIA GESU OPERAIO','FIDENE','CASAL BIANCO','PETRIANA','T CALCIO']
URL='https://www.calcioa5live.com/'
def normalize(s):
 s=' '.join(s.upper().replace('Ù','U').replace('’',"'").split())
 return {'GESU OPERAIO':'PARROCCHIA GESU OPERAIO','OLIMPUS TIBURTINA':'OLIMPUS TIBURTINA ROMA','IL CASALE':'IL CASALE DI ROMA'}.get(s,s)
def parse(html,r):
 s=BeautifulSoup(html,'html.parser'); out={}
 for tr in s.select('tr'):
  cells=[normalize(' '.join(x.stripped_strings)) for x in tr.find_all('td',recursive=False)]
  ids=[(i,t) for i,c in enumerate(cells) for t in TEAMS if c==t]
  if len(ids)!=2: continue
  (i,h),(j,a)=ids
  score=' '.join(cells[i+1:j]); m=re.fullmatch(r'\s*(\d+)\s*[-–:]\s*(\d+)\s*',score)
  if not m and not re.fullmatch(r'\s*[-–]\s*',score): continue
  out[h,a]={'round':r,'home':h,'away':a,'score':[int(m[1]),int(m[2])] if m else None}
 return list(out.values())
def fit(games):
 idx={t:i for i,t in enumerate(TEAMS)}; n=len(TEAMS); latest=max(x['round'] for x in games)
 def loss(z):
  v=.8*np.sum(z[2:]**2)+.8*z[1]**2
  for g in games:
   i,j=idx[g['home']],idx[g['away']]; a,b=g['score']; u=z[0]+z[1]+z[2+i]+z[2+n+j]; w=z[0]+z[2+j]+z[2+n+i]; v+=.85**(latest-g['round'])*(np.exp(u)-a*u+np.exp(w)-b*w)
  return v
 result=minimize(loss,np.r_[np.log(3.5),np.zeros(1+2*n)],method='L-BFGS-B',bounds=[(-2,3)]+[(-2,2)]*(1+2*n))
 if not result.success: raise ValueError('Model fitting failed: '+result.message)
 z=result.x
 def predict(g):
  i,j=idx[g['home']],idx[g['away']]; a,b=np.exp([z[0]+z[1]+z[2+i]+z[2+n+j],z[0]+z[2+j]+z[2+n+i]])
  return dict(g,expected=[round(a,2),round(b,2)],prediction=[int(a+.5),int(b+.5)],prob=[round(float(skellam.sf(0,a,b))*100,1),round(float(skellam.pmf(0,a,b))*100,1),round(float(skellam.cdf(-1,a,b))*100,1)],over=[round(float(poisson.sf(k,a+b))*100,1) for k in [5,7,9]])
 return predict,z
def main():
 games=[]
 for r in range(1,27):
  res=requests.get(URL,params={'annoRiferimento':'2026/2027','com':'girone','giornata':r,'id':21},timeout=30,headers={'User-Agent':'RealMatteiStats/1.0'});res.raise_for_status(); g=parse(res.text,r)
  if len(g)!=7 or len({t for x in g for t in [x['home'],x['away']]})!=14: raise ValueError(f'Giornata {r}: struttura fonte inattesa ({len(g)} gare). Dati precedenti conservati.')
  games+=g;time.sleep(.6)
 if len({(g['home'],g['away']) for g in games})!=182: raise ValueError('Calendario duplicato')
 done=[g for g in games if g['score'] is not None]; todo=[g for g in games if g['score'] is None]
 path=pathlib.Path('docs/data.json'); old=json.loads(path.read_text()) if path.exists() else {}
 if len(done)<14 or (not old.get('demo',True) and len(done)<old.get('played',0)): raise ValueError('Dati insufficienti o regressione dei risultati')
 p,z=fit(done); idx={t:i for i,t in enumerate(TEAMS)}; stats={t:[0,0,0,0] for t in TEAMS}
 for g in done:
  a,b=g['score']; h,v=stats[g['home']],stats[g['away']];h[0]+=3 if a>b else int(a==b);v[0]+=3 if b>a else int(a==b);h[1]+=a;h[2]+=b;v[1]+=b;v[2]+=a;h[3]+=1;v[3]+=1
 rng=np.random.default_rng(42); pts=np.tile([stats[t][0] for t in TEAMS],(20000,1)); gd=np.tile([stats[t][1]-stats[t][2] for t in TEAMS],(20000,1)); predicted=[p(g) for g in todo]
 for g in predicted:
  i,j=idx[g['home']],idx[g['away']];a,b=rng.poisson(g['expected'][0],20000),rng.poisson(g['expected'][1],20000);pts[:,i]+=3*(a>b)+(a==b);pts[:,j]+=3*(b>a)+(a==b);gd[:,i]+=a-b;gd[:,j]+=b-a
 order=np.argsort(-(pts*10000+gd+rng.random(pts.shape)),axis=1); table=[dict(team=t,points=stats[t][0],gf=stats[t][1],ga=stats[t][2],played=stats[t][3],projected=round(float(pts[:,i].mean()),1),title=round(float((order[:,0]==i).mean()*100),1),attack=round(float(np.exp(z[2+i])*100)),defense=round(float(np.exp(-z[16+i])*100))) for i,t in enumerate(TEAMS)];table.sort(key=lambda t:(-t['points'],t['ga']-t['gf']))
 for i,t in enumerate(TEAMS):
  row=next(x for x in table if x['team']==t)
  row.update(top3=round(float((order[:,:3]==i).any(axis=1).mean()*100),1),top5=round(float((order[:,:5]==i).any(axis=1).mean()*100),range=[int(np.quantile(pts[:,i],.1)),int(np.quantile(pts[:,i],.9))],position=int(np.argmax(np.bincount(np.where(order==i)[1],minlength=14)))+1)
  previous=next((x for x in old.get('table',[]) if x['team']==t),{})
  row['titleChange']=round(row['title']-previous.get('title',row['title']),1)
 archive=old.get('archive',{}); current=max((g['round'] for g in done),default=1); current=min(current+int(all(g['score'] is not None for g in games if g['round']==current)),26)
 for g in predicted:
  if g['round'] in (current,current+1): archive.setdefault(str(g['round'])+'|'+g['home'],g)
 history=[dict(v,actual=g['score']) for g in done if (v:=archive.get(str(g['round'])+'|'+g['home']))]
 data=dict(demo=False,updated=datetime.datetime.now(ZoneInfo('Europe/Rome')).isoformat(),round=current,table=table,matches=[g for g in predicted if g['round']==current],archive=archive,history=history,played=len(done),games=games,future=predicted,players=[],discipline=[],coverage={'scorers':None,'discipline':None},schema=2);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False));tmp.replace(path)
if __name__=='__main__':main()
