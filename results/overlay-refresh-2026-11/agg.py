import json,glob,statistics as st,sys
import os
U = os.path.dirname(os.path.abspath(__file__))
M=[("prefill_1","prefill_tok_s","prefill 1x32K tok/s",1),("prefill_1","ttft_s","32K cold TTFT s",0),("prefill_8","prefill_tok_s","prefill 8x32K tok/s",1),
   ("decode_c1","out_tok_s","decode c1 tok/s",1),("decode_c16","out_tok_s","decode c16 tok/s",1),("decode_c64","out_tok_s","decode c64 tok/s",1),
   ("decode_c64","tpot_p50_ms","decode c64 TPOT ms",0),("agent_32k_c16","wall_s","agent 32K c16 wall s",0),("agent_96k_c8","wall_s","agent 96K c8 wall s",0)]
arms=sys.argv[1:]
def get(a,t,k):
    v=[]
    for f in sorted(glob.glob(f"{U}/{a}.p*.perf.json")):
        try: v.append(json.load(open(f))[t][k])
        except Exception: pass
    return v
print(f"{'metric':24}"+"".join(f"{a:>26}" for a in arms))
for t,k,n,hi in M:
    row=f"{n:24}"
    for a in arms:
        v=get(a,t,k)
        row+=f"{(f'{st.median(v):.4g} [{min(v):.4g}-{max(v):.4g}]' if v else '-'):>26}"
    print(row)
for a in arms:
    try: print(a,"GSM8K",json.load(open(f"{U}/{a}.gsm8k.json")).get("accuracy"))
    except Exception as e: print(a,"gsm8k?",e)
