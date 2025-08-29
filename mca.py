#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RS MCA/CA fuzzer (Johnson regime, root-of-unity domain).

- Builds Reed–Solomon code over GF(p).
- Generates δ-close prover words (aligned or unaligned).
- Tests:
    * Correlated Agreement (CA) with list decoding
    * Mutual Correlated Agreement (MCA) with list decoding
    * Reports list sizes for each α
- Emits a counterexample when CA_ok and MCA_size < s.

Designed for small n (n ≤ ~12) so brute force is feasible.
"""

import itertools, math, random, json
from typing import List, Tuple, Optional

# ---------------- Field & poly utils ----------------

def poly_eval(coeffs: List[int], x: int, p: int) -> int:
    y=0
    for c in reversed(coeffs):
        y=(y*x+c)%p
    return y

def eval_poly_vector(coeffs: List[int], xs: List[int], p: int) -> List[int]:
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

def gen_instance(xs,p,k,ell,delta,aligned=True):
    """Generate δ-close words, aligned or unaligned flips."""
    n=len(xs)
    fs=[]
    if aligned:
        t=max(1,math.floor(delta*n))
        flips=sorted(random.sample(range(n),t))
        for _ in range(ell):
            coeffs=rand_poly(k,p)
            c=eval_poly_vector(coeffs,xs,p)
            y=c[:]
            for i in flips: y[i]=(y[i]+1+random.randrange(p-1))%p
            fs.append(y)
    else:
        for _ in range(ell):
            t=max(1,math.floor(delta*n))
            flips=sorted(random.sample(range(n),t))
            coeffs=rand_poly(k,p)
            c=eval_poly_vector(coeffs,xs,p)
            y=c[:]
            for i in flips: y[i]=(y[i]+1+random.randrange(p-1))%p
            fs.append(y)
    return fs

# ---------------- List decoding ----------------

def list_decode(y,xs,k,p,delta):
    """Return all codewords agreeing with y on ≥ s coords."""
    n=len(xs); s=math.ceil((1-delta)*n)
    codewords={}
    for T in itertools.combinations(range(n),k):
        coeffs=interpolate_lagrange([xs[i] for i in T],[y[i] for i in T],k,p)
        cw=tuple(eval_poly_vector(coeffs,xs,p))
        agree=sum(cw[i]==y[i] for i in range(n))
        codewords[cw]=max(codewords.get(cw,0),agree)
    good=[cw for cw,a in codewords.items() if a>=s]
    return s,good

# ---------------- Agreement ----------------

def gen_linear_combo(fs,alpha,p):
    ell=len(fs); n=len(fs[0])
    weights=[pow(alpha,j,p) for j in range(ell)]
    return [(sum(weights[j]*fs[j][i] for j in range(ell))%p) for i in range(n)]

def correlated_agreement(xs,fs,k,p,alphas,delta):
    """CA with list decoding: record list size for each α."""
    good=0; records=[]
    s=math.ceil((1-delta)*len(xs))
    for a in alphas:
        combo=gen_linear_combo(fs,a,p)
        _,good_list=list_decode(combo,xs,k,p,delta)
        list_size=len(good_list)
        records.append({"alpha":a,"list_size":list_size})
        if list_size>0: good+=1
    return good,records


def mutual_agreement(xs,fs,k,p,delta):
    """MCA with list decoding: search for common S across provers."""
    n=len(xs); ell=len(fs)
    s=math.ceil((1-delta)*n)
    best=-1; bestS=None
    # try all k-subsets for each prover, but compare common matches
    for T in itertools.combinations(range(n),k):
        candidates=[]
        for j in range(ell):
            _,good_list=list_decode(fs[j],xs,k,p,delta)
            if not good_list: 
                candidates.append([])
            else:
                candidates.append(good_list)
        # intersection of agreement positions
        match=[True]*n
        for j in range(ell):
            if not candidates[j]: 
                match=[False]*n; break
            # pick one codeword arbitrarily from the list (heuristic)
            cw=candidates[j][0]
            for i in range(n):
                if match[i] and cw[i]!=fs[j][i]:
                    match[i]=False
        size=sum(match)
        if size>best:
            best=size; bestS=[i for i in range(n) if match[i]]
    return best,bestS

# ---------------- Runner ----------------

def run(p=13,n=6,k=2,ell=2,tries=10,aligned=False,alphas=None,seed=None):
    if seed is not None:
        random.seed(seed)
    xs=root_of_unity_domain(p,n)
    rho=k/n
    delta=1-1*math.sqrt(rho)  # Johnson radius
    if alphas is None:
        alphas=list(range(min(p,16)))
    s=math.ceil((1-delta)*n)
    for t in range(tries):
        fs=gen_instance(xs,p,k,ell,delta,aligned)
        # CA
        good,records=correlated_agreement(xs,fs,k,p,alphas,delta)
        CA_ok=(good>=1)  # strict: ∃ α with nonempty list
        # MCA
        MCA_size,MCA_S=mutual_agreement(xs,fs,k,p,delta)
        result={
            "trial":t,
            "params":{"p":p,"n":n,"k":k,"ell":ell,"rho":rho,"delta":delta,"s":s,"aligned":aligned},
            "fs":fs,
            "CA":{"ok":CA_ok,"good":good,"records":records},
            "MCA":{"size":MCA_size,"S":MCA_S}
        }
        print(json.dumps(result,indent=2))
        if MCA_size<s:
            print(f">> COUNTEREXAMPLE: CA holds but MCA fails (MCA_size={MCA_size} < s={s})")
            return result
    print("no counterexample found")
    return None

# ---------------- CLI ----------------

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--p",type=int,default=13)
    ap.add_argument("--n",type=int,default=6)
    ap.add_argument("--k",type=int,default=2)
    ap.add_argument("--ell",type=int,default=2)
    ap.add_argument("--tries",type=int,default=10)
    ap.add_argument("--aligned",action="store_true")
    ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--alphas",type=str,default=None)
    args=ap.parse_args()

    def parse_alphas(arg,p):
        if arg is None: return None
        s=arg.strip()
        if "," in s:
            vals=[int(tok)%p for tok in s.split(",") if tok.strip()!=""]
            out=[]; seen=set()
            for v in vals:
                if v not in seen: seen.add(v); out.append(v)
            return out
        N=int(s); 
        if N<=0: return []
        if N>p: N=p
        return list(range(N))

    parsed_alphas=parse_alphas(args.alphas,args.p)

    run(p=args.p,n=args.n,k=args.k,ell=args.ell,tries=args.tries,
        aligned=args.aligned,alphas=parsed_alphas,seed=args.seed)
