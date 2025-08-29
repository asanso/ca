#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RS MCA/CA fuzzer (Johnson regime, root-of-unity domain) with diagnostics.

- δ = 1 - 1.01*sqrt(rho), s = ceil((1-δ)*n)
- δ-close generators (aligned/unaligned) + optional forced multi-candidate case
- CA uses list decoding (Johnson) and records list_size for each α
  • When list_size == 0: adds 'why_no_list' (best agreement witness, deficit, etc.)
- MCA uses a list-based heuristic and reports detailed failure reasons:
  • For the best witness it shows per-prover candidate codeword and its agree indices,
    and the final common set S (and why it fell short of s)

Emits counterexample if CA_ok and MCA_size < s.
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

def make_multi_candidate_word(xs, p, k, s, overlap=0):
    """
    Build a word y that is δ-close to *two* different codewords.
    Choose two random polynomials cA,cB.
    Pick sets SA, SB of size s with |SA ∩ SB| = overlap.
    """
    n = len(xs)
    if 2*s - overlap > n:
        raise ValueError("overlap too small (need 2s - overlap ≤ n)")

    idx = list(range(n))
    random.shuffle(idx)
    SA = set(idx[:s])
    rest = [i for i in idx if i not in SA]
    SB_only = set(rest[:s - overlap])
    overlap_idxs = set(list(SA)[:overlap])
    SB = set(overlap_idxs) | SB_only

    coeffsA = [random.randrange(p) for _ in range(k)]
    coeffsB = [random.randrange(p) for _ in range(k)]
    cA = eval_poly_vector(coeffsA, xs, p)
    cB = eval_poly_vector(coeffsB, xs, p)

    y = [None]*n
    for i in range(n):
        if i in SA:
            y[i] = cA[i]
        elif i in SB:
            y[i] = cB[i]
        else:
            bad = {cA[i], cB[i]}
            r = random.randrange(p)
            while r in bad:
                r = random.randrange(p)
            y[i] = r

    return y, cA, cB, sorted(SA), sorted(SB)

# ---------------- List decoding & diagnostics ----------------

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

def best_agreement_witness(y, xs, k, p):
    """
    Return a best-agreement witness:
    (max_agree, coeffs_mod_p, codeword, S_star, mismatches)
    """
    n=len(xs); best=-1; best_coeffs=None; best_cw=None
    for T in itertools.combinations(range(n),k):
        coeffs=interpolate_lagrange([xs[i] for i in T],[y[i] for i in T],k,p)
        cw=eval_poly_vector(coeffs,xs,p)
        agree=sum(cw[i]==y[i] for i in range(n))
        if agree>best:
            best=agree; best_coeffs=coeffs; best_cw=cw
            if best==n: break
    S_star=[i for i in range(n) if best_cw[i]==y[i]]
    mismatches=[i for i in range(n) if best_cw[i]!=y[i]]
    return best, [c%p for c in best_coeffs], best_cw, S_star, mismatches

# ---------------- CA / MCA (with diagnostics) ----------------

def gen_linear_combo(fs,alpha,p):
    ell=len(fs); n=len(fs[0])
    weights=[pow(alpha,j,p) for j in range(ell)]
    return [(sum(weights[j]*fs[j][i] for j in range(ell))%p) for i in range(n)]

def correlated_agreement(xs,fs,k,p,alphas,delta):
    """
    CA with list decoding: record list size for each α.
    When list_size == 0, include 'why_no_list' diagnostics.
    """
    good=0; records=[]
    s=math.ceil((1-delta)*len(xs))
    for a in alphas:
        combo=gen_linear_combo(fs,a,p)
        _,good_list=list_decode(combo,xs,k,p,delta)
        list_size=len(good_list)
        rec={"alpha":a,"list_size":list_size}
        if list_size==0:
            max_agree, coeffs, cw, S_star, mism = best_agreement_witness(combo, xs, k, p)
            rec["why_no_list"]={
                "threshold_s": s,
                "max_agreement": max_agree,
                "deficit": s - max_agree,
                "best_poly_coeffs_mod_p": coeffs,
                "best_codeword": cw,
                "match_indices_S_star": S_star,
                "mismatch_indices": mism,
                "combo_word": combo
            }
        records.append(rec)
        if list_size>0: good+=1
    return good,records

def mutual_agreement_debug(xs, fs, k, p, delta, s):
    """
    MCA with diagnostics: search for common S across provers.
    Returns (best_size, bestS, debug_dict).
    For each prover j, we pick one list-decoded candidate (heuristic: first),
    record its agree indices with f_j, and compute the intersection.
    """
    n = len(xs); ell = len(fs)
    best = -1; bestS = None; best_records = None

    # We don't actually need to iterate bases T here for list-based MCA;
    # we enumerate each prover's Johnson list once and compute intersections
    # over all choices is expensive; so we use a simple heuristic:
    # pick the first candidate for each prover.
    # (Extendable: try all small combinations if needed.)
    records = []
    empty_list_found = False
    for j in range(ell):
        _, good_list = list_decode(fs[j], xs, k, p, delta)
        if not good_list:
            empty_list_found = True
            records.append({
                "prover": j,
                "has_candidate": False,
                "candidate_codeword": None,
                "agree_indices": [],
                "word": fs[j]
            })
        else:
            cw = good_list[0]
            agree_set = [i for i in range(n) if cw[i] == fs[j][i]]
            records.append({
                "prover": j,
                "has_candidate": True,
                "candidate_codeword": cw,
                "agree_indices": agree_set,
                "word": fs[j]
            })

    # If any prover had empty list, MCA cannot reach s
    if empty_list_found:
        match = []
    else:
        # Intersect agree sets across provers
        match_mask = [True]*n
        for rec in records:
            cw = rec["candidate_codeword"]
            for i in range(n):
                if match_mask[i] and cw[i] != rec["word"][i]:
                    match_mask[i] = False
        match = [i for i in range(n) if match_mask[i]]

    best = len(match)
    bestS = match
    debug = {
        "per_prover": records,
        "reason": (
            f"MCA_size={best} < s={s}. "
            f"At least one prover had empty Johnson list."
            if empty_list_found else
            f"MCA_size={best} < s={s}. "
            f"Intersection of agree sets too small: {bestS}"
        )
    }
    return best, bestS, debug

# ---------------- Runner ----------------

def run(p=13,n=6,k=2,ell=2,tries=10,aligned=False,alphas=None,seed=None,use_multi=False):
    if seed is not None:
        random.seed(seed)
    xs=root_of_unity_domain(p,n)
    rho=k/n
    delta=1-1.01*math.sqrt(rho)  # Johnson regime
    if alphas is None:
        alphas=list(range(min(p,16)))
    s=math.ceil((1-delta)*n)
    for t in range(tries):
        if use_multi:
            # force a multi-candidate word for prover 0
            y,cA,cB,SA,SB=make_multi_candidate_word(xs,p,k,s,overlap=max(0,2*s-n))
            fs=[y]
            # other provers random δ-close words
            fs.extend(gen_instance(xs,p,k,ell-1,delta,aligned))
        else:
            fs=gen_instance(xs,p,k,ell,delta,aligned)

        # CA (Johnson)
        good,records=correlated_agreement(xs,fs,k,p,alphas,delta)
        CA_ok=(good>=1)  # ∃ α with nonempty list

        # MCA (Johnson, with debug)
        MCA_size,MCA_S,MCA_debug=mutual_agreement_debug(xs,fs,k,p,delta,s)

        result={
            "trial":t,
            "params":{"p":p,"n":n,"k":k,"ell":ell,"rho":rho,"delta":delta,"s":s,"aligned":aligned,"use_multi":use_multi},
            "fs":fs,
            "CA":{"ok":CA_ok,"good":good,"records":records},
            "MCA":{"size":MCA_size,"S":MCA_S,"debug":MCA_debug}
        }
        print(json.dumps(result,indent=2))
        if CA_ok and MCA_size<s:
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
    ap.add_argument("--use-multi",action="store_true",help="Force one prover word to be δ-close to two codewords (multi-candidate)")
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
        N=int(s)
        if N<=0: return []
        if N>p: N=p
        return list(range(N))

    parsed_alphas=parse_alphas(args.alphas,args.p)

    run(p=args.p,n=args.n,k=args.k,ell=args.ell,tries=args.tries,
        aligned=args.aligned,use_multi=args.use_multi,
        alphas=parsed_alphas,seed=args.seed)
