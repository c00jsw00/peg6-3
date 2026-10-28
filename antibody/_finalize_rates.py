import numpy as np, math, json
from pathlib import Path
from bioemu2 import FKCSampler
from run_steered_sampling import SEQ_H, SEQ_L
HERE=Path(Path(__file__).resolve().parent); STEERED=HERE/"steered"; KBT=0.592
def load_windows():
    for pat in ("d*_n*","l*_k*_n*_s*","a*_n*_s*"):
        for d in sorted(STEERED.glob(pat)):
            if not (d/"cv.npz").is_file(): continue
            z=np.load(d/"cv.npz"); yield z["cv1"].astype(float), z["cv2"].astype(int), z["U"].astype(float)
c1_e=np.arange(8.0,152.0,2.0); c2_e=np.arange(0.0,260.0,5.0)
n1,n2=len(c1_e)-1,len(c2_e)-1
prob=np.zeros((n1,n2)); nfr=np.zeros((n1,n2))
for cv1,cv2,U in load_windows():
    u=U-U.max(); w=np.exp(u)
    h,_,_=np.histogram2d(cv1,cv2,weights=w,bins=(c1_e,c2_e)); nh,_,_=np.histogram2d(cv1,cv2,bins=(c1_e,c2_e))
    prob+=h/(w.sum()+1e-12); nfr+=nh
prob/=prob.sum(); pmf=-KBT*np.log(prob+1e-15); pmf[prob<=1e-15]=np.nan
med=float(np.nanmedian(pmf)); pmf=np.where(np.isnan(pmf),med,pmf)
g1=0.5*(c1_e[:-1]+c1_e[1:]); g2=0.5*(c2_e[:-1]+c2_e[1:])
def region_mask(cv1lo,cv1hi,cv2lo,cv2hi):
    m=np.zeros((n1,n2),bool)
    for i in range(n1):
        if cv1lo<=g1[i]<=cv1hi:
            for j in range(n2):
                if cv2lo<=g2[j]<=cv2hi: m[i,j]=True
    return m
free  = min(pmf[region_mask(90,150,0,15)].ravel())
approach = min(pmf[region_mask(55,75,0,8)].ravel())
gate  = min(pmf[region_mask(40,62,15,72)].ravel())
gate_real = np.where(region_mask(40,62,15,72) & (nfr>=3))[0]
print("free plateau      : %.2f" % free)
print("approach (entry)  : %.2f" % approach)
print("cleft-entrance gate (cv1 40-62, cv2 15-72) min = %.2f, real cells=%d" % (gate, len(gate_real)))
print("peak gate (cv1 45-55, cv2 18-45) :")
m=region_mask(45,55,18,45)
vals=sorted([(pmf[i,j],nfr[i,j],g1[i],g2[j]) for i in range(n1) for j in range(n2) if m[i,j]], key=lambda x:-x[0])[:6]
for v in vals: print("  PMF=%.2f frames=%d cv1=%.0f cv2=%.0f" % (v[0],v[1],v[2],v[3]))
peak=max(pmf[i,j] for i in range(n1) for j in range(n2) if m[i,j])
print("peak gate PMF = %.2f" % peak)
dgo_gate_free  = peak - free
dgo_gate_appr  = peak - approach
def kon(dg): return 6.21e12*math.exp(-dg/KBT)
print("\n=== association barrier ===")
print("  dG_on‡ (gate - free)    = %.2f kcal/mol -> k_on = %.3e M^-1/s" % (dgo_gate_free, kon(dgo_gate_free)))
print("  dG_on‡ (gate - approach) = %.2f kcal/mol -> k_on = %.3e M^-1/s" % (dgo_gate_appr, kon(dgo_gate_appr)))
print("\n  experimental: k_on=%.4e -> dG_on‡=%.2f" % (1.1e4, KBT*math.log(6.21e12/1.1e4)))
k_off = 0.066  # floor-limited
Kd = k_off / kon(dgo_gate_appr)
print("\n  K_D = k_off/k_on = %.3e M  (k_off=%.3f s^-1)" % (Kd, k_off))
print("  experimental K_D = 2.04e-7 M")
res={"n_windows_used":len(list(load_windows())),"free_pmf":round(free,2),"approach_pmf":round(approach,2),
     "gate_min_pmf":round(gate,2),"gate_peak_pmf":round(peak,2),"gate_real_cells":int(len(gate_real)),
     "dG_on_gate_free":round(dgo_gate_free,2),"dG_on_gate_approach":round(dgo_gate_appr,2),
     "k_on_gate_free_M_s":float(kon(dgo_gate_free)),"k_on_gate_approach_M_s":float(kon(dgo_gate_appr)),
     "k_off_s":k_off,"K_D_M":float(Kd),"experimental":{"k_on_M_s":1.1e4,"k_off_s":2.25e-3,"K_D_M":2.04e-7}}
out=HERE/"pmf_ladder_final"/"association_barrier.json"
out.write_text(json.dumps(res,indent=2,default=float))
print("\nSaved",out)
