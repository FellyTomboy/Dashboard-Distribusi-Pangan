#!/usr/bin/env python3
"""
Graph Jaringan Distribusi — Bawang Merah
=========================================
Analisis graf untuk jaringan distribusi antar kabupaten/kota di Jawa Timur.

4.3 Konstruksi Graf
   - Node: 38 kab/kota dengan atribut koordinat & status harga
   - Edge: SPARSE — hanya kab dalam radius ≤120 km jalan
   - Bobot edge: kombinasi jarak jalan + disparitas harga
   - Floyd-Warshall: all-pairs shortest paths

4.4 Analisis Jaringan
   - Degree centrality: node dengan koneksi terbanyak
   - Betweenness centrality: node sebagai jembatan/intermediasi
   - Hub identification: Surabaya sebagai hub utama
   - Studi kasus rute konkret: Nganjuk → Sampang via Surabaya

Data: test_predictions_bawang_merah.csv, koordinat_jatim.csv
Output: 'bawang merah/distribusi/output/'
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta
from collections import defaultdict
from itertools import combinations
from math import radians, sin, cos, sqrt, asin

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI
# ============================================================
KOMODITAS = "Bawang Merah"
KOMODITAS_TAG = "bawang_merah"
PRED_PATH = "bawang merah/training/test_predictions_bawang_merah.csv"
KOORD_PATH = "Data Lain-Lain/koordinat_jatim.csv"
OSRM_PATH = "Data Lain-Lain/jarak_waktu_tempuh_osrm.csv"
OUTPUT_DIR = "bawang merah/distribusi/output"

# --- SPARSE GRAPH ---
MAX_EDGE_KM = 120           # hanya koneksi dalam radius 120 km jalan
ROAD_FACTOR = 1.3           # fallback haversine -> jalan
ALPHA_DIST = 0.6
BETA_PRICE = 0.4

# Bobot prioritas jalan — edge menuju/dari hub besar diperkecil bobotnya
HIGHWAY_BONUS = {
    'Kota Surabaya': 0.70,
    'Kabupaten Gresik': 0.85,
    'Kabupaten Sidoarjo': 0.85,
    'Kota Malang': 0.90,
    'Kota Kediri': 0.90,
}

# Threshold klasifikasi — TIDAK DIGUNAKAN LAGI
# (digantikan oleh quantile-based classification dari dashboard.py)
# THRESHOLD_LOWER = -0.052
# THRESHOLD_UPPER = 0.0525

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#f8f9fa','font.size':10,'axes.titlesize':13,'axes.labelsize':11})


# ============================================================
# FUNGSI BANTU
# ============================================================
def normalise_nama(nama):
    return ' '.join(nama.strip().upper().split())

def haversine(lat1, lon1, lat2, lon2):
    R=6371.0; dlat,dlon=radians(lat2-lat1),radians(lon2-lon1)
    a=sin(dlat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R*2*asin(sqrt(a))

def road_dist(la1,lo1,la2,lo2):
    return haversine(la1,lo1,la2,lo2)*ROAD_FACTOR


def floyd_warshall(n, adj):
    """All-pairs shortest path. adj[i]=list of (j,weight)."""
    dist=np.full((n,n),np.inf); nxt=np.full((n,n),-1,dtype=int)
    for i in range(n):
        dist[i,i]=0; nxt[i,i]=i
        for j,w in adj[i]:
            dist[i,j]=w; nxt[i,j]=j
    for k in range(n):
        for i in range(n):
            if dist[i,k]==np.inf: continue
            for j in range(n):
                nd=dist[i,k]+dist[k,j]
                if nd<dist[i,j]-1e-12:
                    dist[i,j]=nd; nxt[i,j]=nxt[i,k]
    return dist, nxt

def reconstruct(nxt,i,j):
    if nxt[i,j]==-1: return []
    p=[i]
    while i!=j: i=nxt[i,j]; p.append(i)
    return p


# ============================================================
# 1. LOAD DATA
# ============================================================
print("="*60)
print(f"  Graf Jaringan Distribusi - {KOMODITAS}")
print("="*60)

print("\n[1] Memuat data...")
df_pred=pd.read_csv(PRED_PATH, parse_dates=['forecast_date'])
df_koord=pd.read_csv(KOORD_PATH)

# Load OSRM distance matrix jika tersedia
df_osrm=None
osrm_lookup={}
if os.path.exists(OSRM_PATH):
    df_osrm=pd.read_csv(OSRM_PATH)
    for _,r in df_osrm.iterrows():
        a=normalise_nama(r['kab_asal']);b=normalise_nama(r['kab_tujuan'])
        if pd.notna(r['jarak_km']):
            osrm_lookup[(a,b)]=r['jarak_km']
            osrm_lookup[(b,a)]=r['jarak_km']
    print(f"     OSRM: {len(df_osrm)} pasangan jarak jalan tersedia")
else:
    print(f"     OSRM: file tidak ditemukan ({OSRM_PATH}), fallback ke haversine")

# Filter incomplete horizon
n0=len(df_pred)
hc=df_pred.groupby(['kab_kota','forecast_date'])['horizon'].nunique().reset_index()
cd=hc[hc['horizon']==7]
df_pred=df_pred.merge(cd[['kab_kota','forecast_date']], on=['kab_kota','forecast_date'], how='inner')
print(f"     Filter horizon: {n0} -> {len(df_pred)} ({(1-len(df_pred)/n0)*100:.1f}% incomplete dropped)")

print(f"     {df_pred['kab_kota'].nunique()} wilayah, {df_pred['forecast_date'].nunique()} hari, "
      f"horizon {df_pred['horizon'].min()}-{df_pred['horizon'].max()}")

# Koordinat
koord_map={}
for _,r in df_koord.iterrows():
    koord_map[normalise_nama(r['nama'])]=(r['latitude'],r['longitude'])
name_coord_map={}
for nama in df_pred['kab_kota'].unique():
    k=normalise_nama(nama)
    if k in koord_map: name_coord_map[nama]=koord_map[k]
    else:
        for pfx in ('KABUPATEN ','KOTA '):
            if k.startswith(pfx):
                sk=k[len(pfx):]
                for kk,vv in koord_map.items():
                    if kk==sk: name_coord_map[nama]=vv; break
print(f"     Tercocokkan: {len(name_coord_map)}/{df_pred['kab_kota'].nunique()}")

kab_list=sorted(df_pred['kab_kota'].unique())
kab_to_idx={k:i for i,k in enumerate(kab_list)}
n=len(kab_list)

df_pred['lat']=df_pred['kab_kota'].map(lambda x:name_coord_map.get(x,(None,None))[0])
df_pred['lon']=df_pred['kab_kota'].map(lambda x:name_coord_map.get(x,(None,None))[1])


# ============================================================
# 2. KLASIFIKASI
# ============================================================
print("\n[2] Klasifikasi harga (quantile 25/75)...")
rp=df_pred.groupby(['forecast_date','horizon'])['prediction'].mean().reset_index()
rp.columns=['forecast_date','horizon','rata_provinsi']
df_pred=df_pred.merge(rp,on=['forecast_date','horizon'],how='left')
df_pred['pct_vs_prov']=(df_pred['prediction']-df_pred['rata_provinsi'])/df_pred['rata_provinsi']

# Quantile-based classification (robust, selalu menghasilkan ~9 sumber & ~9 target)
chunks = []
for (dt, h), grp in df_pred.groupby(['forecast_date', 'horizon']):
    g = grp.copy()
    p25 = g["prediction"].quantile(0.25)
    p75 = g["prediction"].quantile(0.75)
    g["status"] = g["prediction"].apply(
        lambda x: "Harga Rendah" if x <= p25 else
                  ("Harga Tinggi" if x >= p75 else "Harga Rata-rata")
    )
    chunks.append(g)
df_pred = pd.concat(chunks, ignore_index=True)

ra=df_pred.groupby(['forecast_date','horizon'])['actual'].mean().reset_index()
ra.columns=['forecast_date','horizon','rata_aktual']
df_pred=df_pred.merge(ra,on=['forecast_date','horizon'],how='left')
df_pred['pct_vs_prov_act']=(df_pred['actual']-df_pred['rata_aktual'])/df_pred['rata_aktual']

# Classification for actual with quantile
chunks_act = []
for (dt, h), grp in df_pred.groupby(['forecast_date', 'horizon']):
    g = grp.copy()
    p25 = g["actual"].quantile(0.25)
    p75 = g["actual"].quantile(0.75)
    g["status_actual"] = g["actual"].apply(
        lambda x: "Harga Rendah" if x <= p25 else
                  ("Harga Tinggi" if x >= p75 else "Harga Rata-rata")
    )
    chunks_act.append(g)
df_pred = pd.concat(chunks_act, ignore_index=True)
print(f"     Status prediksi: {df_pred['status'].value_counts().to_dict()}")
print(f"     Status aktual:   {df_pred['status_actual'].value_counts().to_dict()}")


# ============================================================
# 3. SPARSE GRAPH + FLOYD-WARSHALL
# ============================================================
print(f"\n[3] Membangun sparse graph (MAX_EDGE={MAX_EDGE_KM} km)...")

# Matriks jarak fisik
dist_km=np.zeros((n,n))
adj=[[] for _ in range(n)]
edge_count=0

# Price diff matrix
pd_mat=np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        ki,kj=kab_list[i],kab_list[j]
        pi=df_pred[df_pred['kab_kota']==ki]['prediction'].values
        pj=df_pred[df_pred['kab_kota']==kj]['prediction'].values
        if len(pi)>0 and len(pj)>0:
            m=min(len(pi),len(pj))
            pd_mat[i,j]=pd_mat[j,i]=np.mean(np.abs(pi[:m]-pj[:m]))

for i in range(n):
    for j in range(i+1,n):
        ki,kj=kab_list[i],kab_list[j]
        
        # Gunakan OSRM jika tersedia, fallback ke haversine
        key=(normalise_nama(ki), normalise_nama(kj))
        if key in osrm_lookup:
            d_km=osrm_lookup[key]
        else:
            la1,lo1=name_coord_map[ki]; la2,lo2=name_coord_map[kj]
            d_km=road_dist(la1,lo1,la2,lo2)
        
        dist_km[i,j]=dist_km[j,i]=d_km

        # SPARSE: hanya edge dalam radius
        if d_km<=MAX_EDGE_KM:
            w=ALPHA_DIST*(d_km/500)+BETA_PRICE*(pd_mat[i,j]/50000)
            w=max(w,0.001)
            # Prioritas jalan: edge menuju/dari hub besar mendapat diskon bobot
            bonus=min(HIGHWAY_BONUS.get(ki,1.0), HIGHWAY_BONUS.get(kj,1.0))
            w*=bonus
            adj[i].append((j,w))
            adj[j].append((i,w))
            edge_count+=1

# Floyd-Warshall
dist_fw, nxt=floyd_warshall(n,adj)

# Jarak fisik shortest path
dist_sp=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if i==j: continue
        path=reconstruct(nxt,i,j)
        if len(path)>=2:
            total=sum(dist_km[a,b] for a,b in zip(path[:-1],path[1:]))
            dist_sp[i,j]=total
        else:
            dist_sp[i,j]=dist_km[i,j]

# Degree (jumlah edge per node)
degrees=[len(adj[i]) for i in range(n)]

# Reachable pairs
# Pasangan reachable (eksklusi diagonal / self-pair)
reachable=np.sum(dist_fw<np.inf) - n  # kurangi n self-pair
reachable/=(n*(n-1)) if n>1 else 1

print(f"     Node: {n}")
print(f"     Edge: {edge_count} (density: {edge_count/(n*(n-1)/2)*100:.1f}%)")
print(f"     Rata-rata degree: {np.mean(degrees):.1f}")
print(f"     Pasangan reachable: {reachable:.1%}")
print(f"     Rata-rata jarak langsung: {dist_km.sum()/(n*(n-1)):.0f} km")
print(f"     Rata-rata shortest path: {dist_sp[dist_sp>0].mean():.0f} km")


# ============================================================
# 4. CENTRALITY ANALYSIS
# ============================================================
print("\n[4] Centrality analysis...")

# Degree centrality (normalized)
deg_centrality={kab_list[i]:degrees[i]/(n-1) for i in range(n)}

# Betweenness centrality (approximated: fraction of shortest paths passing through node)
betweenness={k:0.0 for k in kab_list}
for i in range(n):
    for j in range(i+1,n):
        if dist_fw[i,j]==np.inf: continue
        path=reconstruct(nxt,i,j)
        for node in path[1:-1]:  # exclude source and target
            betweenness[kab_list[node]]+=1.0

# Normalize
total_paths=sum(betweenness.values())
if total_paths>0:
    for k in betweenness: betweenness[k]/=total_paths

# Top hubs
sorted_deg=sorted(deg_centrality.items(), key=lambda x:-x[1])
sorted_bet=sorted(betweenness.items(), key=lambda x:-x[1])

print("     Top 5 Degree Centrality (hub):")
for k,v in sorted_deg[:5]:
    sn=k.replace('Kabupaten ','').replace('Kota ','')
    print(f"       {sn:20s} degree={v:.3f}")

print("     Top 5 Betweenness Centrality (jembatan):")
for k,v in sorted_bet[:5]:
    sn=k.replace('Kabupaten ','').replace('Kota ','')
    print(f"       {sn:20s} betweenness={v:.3f}")


# ============================================================
# 5. DYNAMIC ROUTING PER DATE + HORIZON
# ============================================================
print("\n[5] Dynamic routing per date+horizon (prediksi harga -> bobot edge)...")

dates=sorted(df_pred['forecast_date'].unique())
check_horizons=[1,3,7]
all_recs=[]  # all recommendations across all dates
LEBARAN_EXAMPLE=pd.Timestamp('2024-04-05')  # H-5 Lebaran, for case study
lebanon_recs=[]  # recommendations specifically around Lebaran

for dt in dates:
    for h in check_horizons:
        df_day=df_pred[(df_pred['forecast_date']==dt)&(df_pred['horizon']==h)]
        if len(df_day)!=n: continue  # need all 38 regions

        pred_dict=df_day.set_index('kab_kota')['prediction'].to_dict()
        rata_prov=df_day['rata_provinsi'].iloc[0]

        # Build dynamic adjacency for THIS date+horizon
        adj_dyn=[[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                d_km=dist_km[i,j]
                if d_km>MAX_EDGE_KM: continue
                ki,kj=kab_list[i],kab_list[j]
                pi,pj=pred_dict.get(ki,0),pred_dict.get(kj,0)
                selisih=abs(pi-pj)
                # Bobot: jarak + INVERSE selisih harga
                # selisih besar -> bobot kecil -> prioritas tinggi
                w=ALPHA_DIST*(d_km/500)+BETA_PRICE*(1/(selisih+1))
                w=max(w,0.001)
                bonus=min(HIGHWAY_BONUS.get(ki,1.0),HIGHWAY_BONUS.get(kj,1.0))
                w*=bonus
                adj_dyn[i].append((j,w)); adj_dyn[j].append((i,w))

        # Floyd-Warshall untuk graf dinamis hari ini
        dist_dyn,nxt_dyn=floyd_warshall(n,adj_dyn)

        # Sumber (harga rendah) dan target (harga tinggi) — quantile-based
        vals = np.array(list(pred_dict.values()))
        p25 = np.percentile(vals, 25)
        p75 = np.percentile(vals, 75)
        sumber=[k for k in kab_list if pred_dict.get(k,0) <= p25]
        target=[k for k in kab_list if pred_dict.get(k,0) >= p75]

        if not sumber or not target: continue

        # Untuk setiap target, cari sumber optimal via Dijkstra dinamis
        matched=set()
        for kab_t in target:
            if kab_t in matched: continue
            ti=kab_to_idx[kab_t]
            best_si=min(sumber,key=lambda s:dist_dyn[ti,kab_to_idx[s]] if dist_dyn[ti,kab_to_idx[s]]!=np.inf else 9999)
            si=kab_to_idx[best_si]
            if si==ti or dist_dyn[ti,si]==np.inf: continue
            matched.add(kab_t)
            path=reconstruct(nxt_dyn,si,ti)
            path_names=[kab_list[p].replace('Kabupaten ','').replace('Kota ','') for p in path]
            rec={
                'tanggal':dt,'horizon':h,
                'sumber':best_si,'target':kab_t,
                'jalur':path_names,'n_hop':len(path)-1,
                'jarak_km':dist_km[ti,si],
                'harga_sumber':pred_dict.get(best_si,0),
                'harga_target':pred_dict.get(kab_t,0),
                'selisih_harga':abs(pred_dict.get(kab_t,0)-pred_dict.get(best_si,0)),
            }
            all_recs.append(rec)
            if LEBARAN_EXAMPLE-timedelta(days=3)<=dt<=LEBARAN_EXAMPLE+timedelta(days=3):
                lebanon_recs.append(rec)

print(f"     Total rekomendasi: {len(all_recs)}")
print(f"     Rekomendasi sekitar Lebaran 2024: {len(lebanon_recs)}")

# Find a concrete example from Lebaran period
contoh=None
for rec in all_recs:
    if rec['tanggal']==LEBARAN_EXAMPLE and rec['horizon']==3 and rec['n_hop']>=2:
        contoh=rec
        break
if not contoh and lebanon_recs:
    contoh=max(lebanon_recs,key=lambda r:r['selisih_harga'])

if contoh:
    print(f"\n     Contoh rekomendasi (Lebaran {contoh['tanggal'].date()}, H+{contoh['horizon']}):")
    print(f"       Sumber (harga rendah): {contoh['sumber']} (Rp {contoh['harga_sumber']:,.0f})")
    print(f"       Target (harga tinggi): {contoh['target']} (Rp {contoh['harga_target']:,.0f})")
    print(f"       Selisih: Rp {contoh['selisih_harga']:,.0f}")
    print(f"       Jalur: {' -> '.join(contoh['jalur'])}")
    print(f"       Jarak: {contoh['jarak_km']:.0f} km, {contoh['n_hop']} hop")


# ============================================================
# 6. VISUALISASI
# ============================================================
print("\n[6] Visualisasi...")

dominant=df_pred.groupby('kab_kota')['status'].apply(lambda x:x.value_counts().index[0]).to_dict()
cm_node={'Harga Rendah':'#1a9641','Harga Rata-rata':'#f7f7f7','Harga Tinggi':'#d73027'}

# --- FIGURE 1: SPARSE NETWORK GRAPH ---
print("     Figure 1: Sparse network graph...")
fig1,ax1=plt.subplots(figsize=(14,12))

for i in range(n):
    for j,w in adj[i]:
        if i<j: continue
        ki,kj=kab_list[i],kab_list[j]
        lo1,la1=name_coord_map[ki][1],name_coord_map[ki][0]
        lo2,la2=name_coord_map[kj][1],name_coord_map[kj][0]
        alpha=1.0-min(w/2,0.8)
        ax1.plot([lo1,lo2],[la1,la2],color='#2c3e50',lw=0.4,alpha=alpha,zorder=1)

for k,(lat,lon) in name_coord_map.items():
    c=cm_node.get(dominant.get(k,'Harga Rata-rata'),'#ccc')
    e='white' if dominant.get(k) in ('Harga Rendah','Harga Tinggi') else '#999'
    ax1.scatter(lon,lat,c=c,edgecolors=e,lw=0.8,s=140,zorder=2,alpha=0.9)
    sn=k.replace('Kabupaten ','').replace('Kota ','')
    ax1.annotate(sn,(lon,lat),xytext=(3,3),textcoords='offset points',fontsize=5,alpha=0.8)

ax1.set_xlim(110.5,115.0); ax1.set_ylim(-8.9,-6.7); ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title(f'Graf Jaringan Distribusi {KOMODITAS} (Sparse)\n'
              f'{n} Node | {edge_count} Edge (radius <= {MAX_EDGE_KM} km) | '
              f'Density {edge_count/(n*(n-1)/2)*100:.1f}%',
              fontsize=13,fontweight='bold')

leg1=[Patch(facecolor='#1a9641',label='Harga Rendah (sumber)'),
      Patch(facecolor='#f7f7f7',edgecolor='#999',label='Harga Rata-rata'),
      Patch(facecolor='#d73027',label='Harga Tinggi (target)'),
      Line2D([0],[0],color='#2c3e50',lw=1,alpha=0.5,label=f'Edge (<= {MAX_EDGE_KM}km)')]
ax1.legend(handles=leg1,loc='lower left',fontsize=8,framealpha=0.8)
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR,'01_sparse_network.png'),dpi=200,bbox_inches='tight')
plt.close(fig1); print("       OK -> 01_sparse_network.png")


# --- FIGURE 2: CENTRALITY ANALYSIS ---
print("     Figure 2: Centrality analysis...")
fig2,(ax2a,ax2b)=plt.subplots(1,2,figsize=(18,8))

# Panel A: Degree centrality map
for feat_name,(lat,lon) in name_coord_map.items():
    dc=deg_centrality.get(feat_name,0)
    c=plt.cm.YlOrRd(dc/max(deg_centrality.values()))
    ax2a.scatter(lon,lat,c=[c],s=80+dc*2000,edgecolors='#333',lw=0.5,zorder=2,alpha=0.8)
    sn=feat_name.replace('Kabupaten ','').replace('Kota ','')
    ax2a.annotate(sn,(lon,lat),xytext=(3,3),textcoords='offset points',fontsize=6,alpha=0.8)

# Plot edges as background
for i in range(n):
    for j,w in adj[i]:
        if i>=j: continue
        ki,kj=kab_list[i],kab_list[j]
        lo1,la1=name_coord_map[ki][1],name_coord_map[ki][0]
        lo2,la2=name_coord_map[kj][1],name_coord_map[kj][0]
        ax2a.plot([lo1,lo2],[la1,la2],color='#ccc',lw=0.3,zorder=1)

ax2a.set_xlim(110.5,115.0); ax2a.set_ylim(-8.9,-6.7); ax2a.set_aspect('equal')
ax2a.axis('off')
ax2a.set_title('Degree Centrality\n(makin besar lingkaran = makin banyak koneksi)',fontsize=12,fontweight='bold')

# Panel B: Betweenness centrality bar chart (top 10)
top10_bet=sorted_bet[:10]
labels_bet=[k.replace('Kabupaten ','').replace('Kota ','') for k,v in top10_bet]
vals_bet=[v for k,v in top10_bet]
colors_bet=plt.cm.Set2(np.linspace(0,1,10))
bars=ax2b.barh(range(len(labels_bet)),vals_bet,color=colors_bet,edgecolor='white')
ax2b.set_yticks(range(len(labels_bet))); ax2b.set_yticklabels(labels_bet,fontsize=9)
ax2b.invert_yaxis()
for bar,v in zip(bars,vals_bet):
    ax2b.text(bar.get_width()+0.001,bar.get_y()+bar.get_height()/2,f'{v:.3f}',
              va='center',fontsize=8)
ax2b.set_xlabel('Betweenness Centrality',fontsize=11,fontweight='bold')
ax2b.set_title('Top 10 Node Berdasarkan Betweenness\n(peran sebagai jembatan/jalur distribusi)',
               fontsize=12,fontweight='bold')
ax2b.set_xlim(0,max(vals_bet)*1.3)

fig2.suptitle(f'Centrality Analysis - {KOMODITAS}',fontsize=14,fontweight='bold',y=0.98)
plt.tight_layout(rect=[0,0,1,0.95])
fig2.savefig(os.path.join(OUTPUT_DIR,'02_centrality.png'),dpi=200,bbox_inches='tight')
plt.close(fig2); print("       OK -> 02_centrality.png")


# --- FIGURE 3: NETWORK EFFICIENCY ---
print("     Figure 3: Network efficiency...")
fig3,(ax3a,ax3b)=plt.subplots(1,2,figsize=(16,7))

# Panel A: Direct vs Network distance scatter
direct_all=[]; network_all=[]
for i in range(n):
    for j in range(i+1,n):
        direct_all.append(dist_km[i,j])
        network_all.append(dist_sp[i,j])

ax3a.scatter(direct_all,network_all,s=5,alpha=0.4,color='#2c3e50')
ax3a.plot([0,400],[0,400],'r--',lw=1,alpha=0.5,label='y=x (sama)')
ax3a.set_xlabel('Jarak Langsung (km)',fontsize=11,fontweight='bold')
ax3a.set_ylabel('Jarak via Jaringan (km)',fontsize=11,fontweight='bold')
ax3a.set_title('Jarak Langsung vs Jaringan\n(titik di atas garis = jaringan lebih panjang)',
               fontsize=12,fontweight='bold')
ax3a.legend(fontsize=9)
ax3a.set_xlim(0,400); ax3a.set_ylim(0,400)

# Panel B: Degree distribution
deg_counts=pd.Series(degrees).value_counts().sort_index()
ax3b.bar(deg_counts.index,deg_counts.values,color='#3498db',edgecolor='white',width=0.8)
ax3b.set_xlabel('Degree (jumlah edge per node)',fontsize=11,fontweight='bold')
ax3b.set_ylabel('Jumlah Node',fontsize=11,fontweight='bold')
ax3b.set_title(f'Distribusi Degree\nRata-rata: {np.mean(degrees):.1f}, '
               f'Min: {min(degrees)}, Max: {max(degrees)}',fontsize=12,fontweight='bold')
ax3b.set_xticks(range(0,int(deg_counts.index.max())+2,2))

fig3.suptitle(f'Network Efficiency Metrics - {KOMODITAS}',fontsize=14,fontweight='bold',y=0.98)
plt.tight_layout(rect=[0,0,1,0.95])
fig3.savefig(os.path.join(OUTPUT_DIR,'03_network_efficiency.png'),dpi=200,bbox_inches='tight')
plt.close(fig3); print("       OK -> 03_network_efficiency.png")


# --- FIGURE 4: DYNAMIC ROUTE CASE STUDY (hasil dari dynamic routing) ---
print("     Figure 4: Dynamic route case study...")

# Gunakan contoh dari dynamic routing
case=contoh if contoh else (lebanon_recs[0] if lebanon_recs else None)

fig4,(ax4a,ax4b)=plt.subplots(1,2,figsize=(18,9),gridspec_kw={'width_ratios':[1.5,1]})

if case:
    # Panel A: Map with route highlighted
    for i in range(n):
        for j,w in adj[i]:
            if i>=j: continue
            ki,kj=kab_list[i],kab_list[j]
            lo1,la1=name_coord_map[ki][1],name_coord_map[ki][0]
            lo2,la2=name_coord_map[kj][1],name_coord_map[kj][0]
            ax4a.plot([lo1,lo2],[la1,la2],color='#ccc',lw=0.3,zorder=1,alpha=0.5)

    # Highlight the route
    path_names=[kab_list[i].replace('Kabupaten ','').replace('Kota ','')
                for i in range(n)
                if kab_list[i] in case['jalur'] or kab_list[i]==case['sumber'] or kab_list[i]==case['target']]
    # Actually use the stored path
    route_nodes=[case['sumber']]+[' '.join(x.split()) for x in case['jalur'] if x not in (case['sumber'],case['target'])]+[case['target']]
    # Deduplicate while preserving order
    seen=set(); route_nodes_uniq=[]
    for n_ in case['jalur']:
        if n_ not in seen: route_nodes_uniq.append(n_); seen.add(n_)

    path_idx=[kab_to_idx[nm] for nm in route_nodes_uniq if nm in kab_to_idx]
    for a,b in zip(path_idx[:-1],path_idx[1:]):
        ki,kj=kab_list[a],kab_list[b]
        lo1,la1=name_coord_map[ki][1],name_coord_map[ki][0]
        lo2,la2=name_coord_map[kj][1],name_coord_map[kj][0]
        ax4a.plot([lo1,lo2],[la1,la2],color='#e74c3c',lw=2.5,zorder=3,alpha=0.85)
        ax4a.annotate('',xy=(lo2,la2),xytext=(lo1,la1),
                     arrowprops=dict(arrowstyle='->',color='#e74c3c',lw=1.5),zorder=4)

    # All nodes
    for k,(lat,lon) in name_coord_map.items():
        in_route=k in route_nodes_uniq
        if in_route:
            c='#e74c3c'; s=200; e='#c0392b'; lw2=1.5
        else:
            c=cm_node.get(dominant.get(k,'Harga Rata-rata'),'#ccc')
            s=100; e='white'; lw2=0.5
        ax4a.scatter(lon,lat,c=c,edgecolors=e,lw=lw2,s=s,zorder=2,alpha=0.9)
        sn=k.replace('Kabupaten ','').replace('Kota ','')
        fs=9 if in_route else 5
        ax4a.annotate(sn,(lon,lat),xytext=(4,4),textcoords='offset points',
                     fontsize=fs,fontweight='bold' if in_route else 'normal',
                     color='#c0392b' if in_route else '#333')

    ax4a.set_xlim(110.5,115.0); ax4a.set_ylim(-8.9,-6.7); ax4a.set_aspect('equal')
    ax4a.axis('off')
    ax4a.set_title(f'Route Dinamis: {case["jalur"][0]} \u2192 {case["jalur"][-1]}',
                   fontsize=13,fontweight='bold',pad=10)

    # Panel B: Route details
    ax4b.axis('off')
    dt=f"DINAMIC ROUTING CASE STUDY\n{'='*35}\n\n"
    dt+=f" Tanggal: {case['tanggal'].date()}\n"
    dt+=f" Horizon: H+{case['horizon']}\n\n"
    dt+=f" Sumber (harga rendah):\n"
    dt+=f"   {case['sumber']}\n"
    dt+=f"   Rp {case['harga_sumber']:,.0f}\n\n"
    dt+=f" Target (harga tinggi):\n"
    dt+=f"   {case['target']}\n"
    dt+=f"   Rp {case['harga_target']:,.0f}\n\n"
    dt+=f" Selisih harga:\n"
    dt+=f"   Rp {case['selisih_harga']:,.0f}\n\n"
    dt+=f" Jalur:\n"
    for idx,n_ in enumerate(route_nodes_uniq):
        arr=' \u2192 ' if idx<len(route_nodes_uniq)-1 else ''
        dt+=f"   {n_}{arr}\n"
    dt+=f"\n Jarak: {case['jarak_km']:.0f} km\n"
    dt+=f" Hop: {case['n_hop']}\n\n"
    dt+=f"---\nJustifikasi: Prediksi harga H+{case['horizon']}\n"
    dt+=f"mendeteksi disparitas Rp {case['selisih_harga']:,.0f}\n"
    dt+=f"yang membuat rute ini prioritas tinggi."

    ax4b.text(0.05,0.95,dt,transform=ax4b.transAxes,fontsize=10,
             verticalalignment='top',fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5',facecolor='#fef9e7',alpha=0.8))
else:
    ax4a.text(0.5,0.5,'Tidak ada contoh rute yang tersedia',ha='center',va='center',fontsize=14)
    ax4b.axis('off')

fig4.suptitle(f'Dynamic Routing - {KOMODITAS}',fontsize=14,fontweight='bold',y=0.98)
plt.tight_layout(rect=[0,0,1,0.95])
fig4.savefig(os.path.join(OUTPUT_DIR,'04_dynamic_route.png'),dpi=200,bbox_inches='tight')
plt.close(fig4); print("       OK -> 04_dynamic_route.png")


# ============================================================
# 7. SUMMARY
# ============================================================
print("\n"+"="*60)
print("  RINGKASAN ANALISIS JARINGAN")
print("="*60)

print(f"\n  --- Topologi Graf ---")
print(f"  Node: {n}")
print(f"  Edge: {edge_count} (sparse, radius <={MAX_EDGE_KM} km)")
print(f"  Density: {edge_count/(n*(n-1)/2)*100:.1f}%")
print(f"  Rata-rata degree: {np.mean(degrees):.1f}")
print(f"  Pasangan reachable: {reachable:.1%}")
print(f"  Rata-rata jarak langsung: {dist_km.sum()/(n*(n-1)):.0f} km")
print(f"  Rata-rata shortest path: {dist_sp[dist_sp>0].mean():.0f} km")

print(f"\n  --- Centrality ---")
print(f"  Top hubs (degree):")
for k,v in sorted_deg[:3]:
    sn=k.replace('Kabupaten ','').replace('Kota ','')
    print(f"    {sn:20s} degree={v:.3f}")
print(f"  Top jembatan (betweenness):")
for k,v in sorted_bet[:3]:
    sn=k.replace('Kabupaten ','').replace('Kota ','')
    print(f"    {sn:20s} betweenness={v:.3f}")

if case:
    print(f"\n  --- Dynamic Route Example ---")
    print(f"  Sumber: {case['sumber']} -> Target: {case['target']}")
    print(f"  Jalur: {' -> '.join(case['jalur'])} ({case['jarak_km']:.0f} km, {case['n_hop']} hop)")

print(f"\n  Output di {OUTPUT_DIR}/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    sz=os.path.getsize(os.path.join(OUTPUT_DIR,f))/1024
    print(f"    {f:40s} {sz:.0f} KB")
print("  Selesai.")