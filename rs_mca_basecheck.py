#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RS CA/MCA fuzzer with base closeness check
using β from the Johnson regime (Proximity Gaps for Reed–Solomon Codes).

Implements:
  1. Base closeness: each f_j individually δ-close to some RS codeword.
  2. Correlated Agreement (CA): for most α, f*_α is δ-close.
     Success fraction must be ≥ β, where
         β = (k+1)^2 / [ (2*min(η, sqrt(ρ)/20))^7 * q ]
     with ρ = (k+1)/n, η = (1 - sqrt(ρ)) - δ.
  3. MCA: search for a common witness set S of size ≥ s.

Defaults:
  - δ = 1 - sqrt(ρ) (Johnson radius)
  - domain = multiplicative subgroup (root of unity)
"""

import itertools, math, random, json
from typing import List

# ---------------- Field & poly utils ----------------

def poly_eval(coeffs: List[int], x: int, p: int) -> int:
    y=0
    for c in reversed(coeffs):
        y=(y*x+c)%p
    return y

def eval_poly_vector(coeffs: List[int], xs: List[int], p: int)->List[int]:
    return [poly_eval(coeffs,x,p) for x in xs]

def poly_add(a,b,p):
    m=max(len(a),len(b))
    out=[0]*m
    for i in range(m):
        if i<len(a): out[i]=(out[i]+a[i])%p
        if i<len(b): out[i]=(out[i]+b[i])%p
    return out

def poly_mul(a,b,p):
    out=[0]*(len(a)+len(b)-1)
    for i,ai in enumerate(a):
        for j,bj in enumerate(b):
            out[i+j]=(out[i+j]+ai*bj)%p
    return out

def poly_scale(a,s,p):
    return [(ai*s)%p for ai in a]

def interpolate_lagrange(xs,ys,k,p):
    coeffs=[0]*k
    for i in range(k):
        xi,yi=xs[i]%p,ys[i]%p
        num=[1]; denom=1
        for m in range(k):
            if m==i: continue
            num=poly_mul(num,[(-xs[m])%p,1],p)
            denom=(denom*((xi-xs[m])%p))%p
        inv=pow(denom,p-2,p)
        li=poly_scale(num,(yi*inv)%p,p)
        coeffs=poly_add(coeffs,li,p)
    coeffs=(coeffs+[0]*k)[:k]
    return coeffs

# ---------------- Domain ----------------

def primitive_root(p:int)->int:
    if p==2: return 1
    phi=p-1
    factors=[]
    m=phi; d=2
    while d*d<=m:
        while m%d==0:
            factors.append(d); m//=d
        d+=1
    if m>1: factors.append(m)
    for g in range(2,p):
        ok=True
        for q in set(factors):
            if pow(g,phi//q,p)==1:
                ok=False; break
        if ok: return g
    raise RuntimeError("no prim root?")

def root_of_unity_domain(p:int,n:int):
    if (p-1)%n!=0: raise ValueError("n must divide p-1")
    g=primitive_root(p)
    omega=pow(g,(p-1)//n,p)
    xs=[1]
    for i in range(1,n):
        xs.append((xs[-1]*omega)%p)
    return xs

# ---------------- Generators ----------------

def rand_poly(k,p): return [random.randrange(p) for _ in range(k)]

def gen_instance(xs,p,k,ell,mode="patchwork",err_frac=0.45):
    n=len(xs); fs=[]
    if mode=="patchwork":
        for _ in range(ell):
            cA=eval_poly_vector(rand_poly(k,p),xs,p)
            cB=eval_poly_vector(rand_poly(k,p),xs,p)
            idx=list(range(n)); random.shuffle(idx)
            cut=n//2; A=set(idx[:cut])
            y=[cA[i] if i in A else cB[i] for i in range(n)]
            flips=random.sample(range(n),max(1,n//6))
            for i in flips: y[i]=(y[i]+1+random.randrange(p-1))%p
            fs.append(y)
    else:
        raise ValueError("use mode='patchwork'")
    return fs

# ---------------- Agreement checks ----------------

def best_agreement_exact(y,xs,k,p):
    n=len(xs); best=-1
    for T in itertools.combinations(range(n),k):
        coeffs=interpolate_lagrange([xs[i] for i in T],
                                    [y[i] for i in T],k,p)
        cw=eval_poly_vector(coeffs,xs,p)
        agree=sum(cw[i]==y[i] for i in range(n))
        if agree>best: best=agree
    return best

def gen_linear_combo(fs,alpha,p):
    ell=len(fs); n=len(fs[0])
    weights=[pow(alpha,j,p) for j in range(ell)]
    return [(sum(weights[j]*fs[j][i] for j in range(ell)) % p) for i in range(n)]

def check_base_closeness(fs,xs,k,p,s):
    records=[]; ok=True
    for j,f in enumerate(fs):
        agree=best_agreement_exact(f,xs,k,p)
        records.append((j,agree))
        if agree<s: ok=False
    return ok,records

def correlated_agreement(xs,fs,k,p,alphas,s,beta):
    good=0; records=[]
    for a in alphas:
        combo=gen_linear_combo(fs,a,p)
        agree=best_agreement_exact(combo,xs,k,p)
        records.append((a,agree))
        if agree>=s: good+=1
    frac=good/len(alphas)
    return frac>=beta, frac, records

def mutual_agreement(xs,fs,k,p):
    n=len(xs); ell=len(fs)
    best=-1; bestS=None
    for T in itertools.combinations(range(n),k):
        coeffs_list=[]
        for j in range(ell):
            coeffs=interpolate_lagrange([xs[i] for i in T],
                                        [fs[j][i] for i in T],k,p)
            coeffs_list.append(coeffs)
        match=[True]*n
        for j in range(ell):
            cw=eval_poly_vector(coeffs_list[j],xs,p)
            for i in range(n):
                if match[i] and cw[i]!=fs[j][i]:
                    match[i]=False
        size=sum(match)
        if size>best:
            best=size; bestS=[i for i in range(n) if match[i]]
    return best,bestS

# ---------------- Johnson beta ----------------

def compute_beta(n,k,q,delta):
    rho=(k+1)/n
    if not ((1-rho)/2 < delta < 1-math.sqrt(rho)):
        raise ValueError("delta must be in Johnson regime")
    eta=(1-math.sqrt(rho))-delta
    denom=(2*min(eta, math.sqrt(rho)/20))**7 * q
    return (k+1)**2 / denom

# ---------------- Runner ----------------

def run(p=13,n=6,k=1,ell=2,tries=1,delta=None,alphas=None):
    xs=root_of_unity_domain(p,n)
    rho=(k+1)/n
    if delta is None:
        delta=1-math.sqrt(rho) - 0.05  # just inside Johnson radius
    s=math.ceil((1-delta)*n)
    beta=compute_beta(n,k,p,delta)
    if alphas is None:
        alphas=list(range(p))   # test all
    for t in range(tries):
        fs=gen_instance(xs,p,k+1,ell,"patchwork")
        base_ok,base_records=check_base_closeness(fs,xs,k+1,p,s)
        CA_ok,CA_frac,CA_records=correlated_agreement(xs,fs,k+1,p,alphas,s,beta)
        MCA_size,MCA_S=mutual_agreement(xs,fs,k+1,p)
        result={
            "params":{"p":p,"n":n,"k":k,"ell":ell,"rho":rho,"delta":delta,"s":s,"beta":beta},
            "domain":xs,
            "fs":fs,
            "base_closeness":{"ok":base_ok,"records":base_records},
            "CA":{"ok":CA_ok,"frac":CA_frac,"records":CA_records},
            "MCA":{"size":MCA_size,"S":MCA_S}
        }
        print(json.dumps(result,indent=2))
        if base_ok and CA_ok and MCA_size<s:
            print(">> COUNTEREXAMPLE: bases close, CA holds, MCA fails (size",MCA_size,"<",s,")")
            return result
    print("no counterexample found"); return None

if __name__=="__main__":
    run()
