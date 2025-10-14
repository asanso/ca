⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ **WORK IN PROGRESS** ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

# Example Usage

`python3 main.py --max-pairs 1000 --stride2 100 --stride1 100 --delta=0.28y`

**Algorithm: Threshold-count test over a code $C$**

**Inputs.** Code $C$, field $\mathbb{F}$, threshold $\delta$, target $E$.

1. Choose $f_1, f_2$ such that  
   $\Delta(f_1, C) > \delta \;\lor\; \Delta(f_2, C) > \delta$.
2. Set $\mathrm{ctr} \leftarrow 0$.
3. For each $\alpha \in \mathbb{F}$:
   - If $\Delta\\big(f_1 + \alpha f_2,\; C\big) \le \delta$, then set  
     $\mathrm{ctr} \leftarrow \mathrm{ctr} + 1$.
4. Return **win** if $\mathrm{ctr} > E$; otherwise **lose**.

**Notes.** $\Delta(f, C)$ denotes the (relative) distance from $f$ to the nearest codeword in $C$.
