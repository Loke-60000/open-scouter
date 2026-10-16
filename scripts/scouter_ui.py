#!/usr/bin/env python3
"""Scouter UI - chat with gemma-4-12B, monitor per-token emotion, and steer.

A self-contained FastAPI app (transformers backend, 4-bit) that serves one HTML
page. Each generated token is colored by the emotion the model's activation
evokes at that token: hue = emotion family (anger=red, joy=gold, sadness=blue,
fear=purple, calm=green, ...), saturation/opacity = how strongly it is felt.
Steering controls let you push the model toward a concept (the assistant axis,
an emotion, or a persona) with a chosen mode + coefficient, applied to the next
message.

    pip install -e ".[generate]" bitsandbytes
    HF_HOME=~/hf-scratch python scripts/scouter_ui.py            # http://localhost:8800
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scouter.axis import load_axis
from scouter.axis.generate import GenerationConfig, load_model_and_tokenizer
from scouter.emotions import EmotionVectors

OUT = Path(__file__).resolve().parent.parent / "precomputed" / "gemma-4-12b"
MONITOR_LAYER = 42
STEER_LAYER = 24
PERSONAS = ["pirate", "ghost", "vampire", "robot", "oracle", "demon",
            "caveman", "detective", "poet", "makise_kurisu"]

# emotion family -> (hex color, member emotions). Anything unlisted falls back to gray.
FAMILIES: dict[str, tuple[str, list[str]]] = {
    "anger": ("#e8413c", ["angry", "furious", "enraged", "irate", "irritated", "annoyed",
                           "frustrated", "exasperated", "hostile", "outraged", "indignant",
                           "resentful", "bitter", "hateful", "vengeful", "vindictive",
                           "spiteful", "mad", "grumpy", "offended", "insulted"]),
    "disgust": ("#9c8a3e", ["disgusted", "contemptuous", "disdainful", "scornful", "smug"]),
    "joy": ("#f4c20d", ["joyful", "happy", "cheerful", "elated", "ecstatic", "delighted",
                        "excited", "thrilled", "euphoric", "jubilant", "content", "pleased",
                        "optimistic", "hopeful", "hope", "exuberant", "vibrant", "blissful",
                        "playful", "amused", "triumphant", "proud", "stimulated", "eager"]),
    "love": ("#ff6fae", ["loving", "grateful", "thankful", "compassionate", "empathetic",
                         "sympathetic", "kind", "infatuated", "sentimental", "nostalgic"]),
    "sadness": ("#3b7dd8", ["sad", "gloomy", "melancholy", "depressed", "heartbroken",
                            "miserable", "grief_stricken", "lonely", "dispirited", "droopy",
                            "unhappy", "regretful", "remorseful", "sorry", "worthless", "hurt",
                            "resigned", "brooding"]),
    "fear": ("#8b5cf6", ["afraid", "scared", "frightened", "terrified", "anxious", "nervous",
                         "panicked", "worried", "tense", "uneasy", "alarmed", "on_edge",
                         "horrified", "shaken", "rattled", "unnerved", "unsettled", "distressed",
                         "desperate", "trapped", "paranoid", "vigilant", "alert", "disturbed",
                         "overwhelmed", "stressed", "hysterical"]),
    "calm": ("#2bb673", ["calm", "serene", "peaceful", "relaxed", "at_ease", "safe",
                         "satisfied", "fulfilled", "docile", "patient", "reflective",
                         "refreshed", "rejuvenated"]),
    "surprise": ("#17becf", ["surprised", "amazed", "astonished", "shocked", "dumbstruck",
                             "awestruck", "bewildered", "perplexed", "puzzled", "mystified",
                             "disoriented", "stuck", "defiant", "obstinate", "stubborn"]),
    "fatigue": ("#8a90a6", ["sluggish", "sleepy", "tired", "weary", "worn_out", "lazy",
                            "listless", "bored", "indifferent"]),
}
DEFAULT_COLOR = "#8a90a6"


def emotion_color_map(names: list[str]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for fam, (color, members) in FAMILIES.items():
        for m in members:
            out[m] = (fam, color)
    return {n: out.get(n, ("other", DEFAULT_COLOR)) for n in names}


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


class SteerSpec(BaseModel):
    concept: str = "none"          # "none" | "assistant_axis" | "emotion:<n>" | "persona:<n>"
    mode: str = "off"              # "off" | "addition"
    coefficient: float = 0.0       # norm-relative


class GenRequest(BaseModel):
    messages: list[dict]
    steer: SteerSpec = SteerSpec()
    temperature: float = 0.0
    max_new_tokens: int = 80


class Scouter:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model: Any = None
        self.tok: Any = None
        self.ev: Optional[EmotionVectors] = None
        self.axis: Any = None
        self.persona_dirs: dict[str, np.ndarray] = {}
        self.emo_colors: dict[str, tuple[str, str]] = {}

    def load(self) -> None:
        cfg = GenerationConfig(position="last", dtype="bfloat16", device_map="auto",
                               load_in_4bit=True)
        self.model, self.tok = load_model_and_tokenizer(self.model_name, cfg)
        self.model.eval()
        self.ev = EmotionVectors.load(str(OUT / "emotions.npz"))
        self.axis = load_axis(str(OUT / "axis.npz"))
        self.emo_colors = emotion_color_map(self.ev.names)
        self.special_ids = set(self.tok.all_special_ids)
        # unit emotion directions at the monitor layer, as one matrix for fast scoring
        self.emo_names = list(self.ev.names)
        u = np.stack([self.ev.vector(n)[MONITOR_LAYER] for n in self.emo_names]).astype(np.float32)
        self.emoU = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
        shard_dir = OUT / "_ckpt" / "persona_means"
        if shard_dir.exists():
            grand = np.mean(np.stack([np.load(p) for p in shard_dir.glob("*.npy")]), axis=0)
            for name in PERSONAS:
                p = shard_dir / f"character__{name}.npy"
                if p.exists():
                    self.persona_dirs[name] = (np.load(p) - grand).astype(np.float32)
        from steer_matrix import decoder_layers  # local import; scripts/ on path
        self.layers = decoder_layers(self.model)
        # baseline = mean reply-token activation on bland prompts. Subtracting it
        # removes the reply-distribution common-mode (which otherwise makes every
        # neutral token read as "sluggish"/fatigue), so only genuinely emotional
        # tokens light up.
        self.reply_baseline = self._reply_baseline()

    @torch.no_grad()
    def _reply_baseline(self) -> np.ndarray:
        acts = []
        for p in ("Describe a plain wooden chair in an empty room.",
                  "List a few common kitchen items."):
            msgs = [{"role": "user", "content": p}]
            prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            plen = enc["input_ids"].shape[1]
            out = self.model.generate(**enc, max_new_tokens=40, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            res = self.model(out, output_hidden_states=True, use_cache=False)
            hs = res.hidden_states[1:][MONITOR_LAYER][0][plen:]
            acts.append(hs.float().mean(dim=0).cpu().numpy())
        return np.mean(acts, axis=0).astype(np.float32)

    def concepts(self) -> dict:
        return {
            "axis": ["assistant_axis"],
            "emotions": self.ev.names if self.ev else [],
            "personas": list(self.persona_dirs),
        }

    def _unit(self, concept: str, layer: int) -> Optional[np.ndarray]:
        if concept == "assistant_axis":
            return self.axis.unit(layer)
        if concept.startswith("emotion:") and self.ev is not None:
            v = self.ev.vector(concept.split(":", 1)[1])[layer]
        elif concept.startswith("persona:"):
            v = self.persona_dirs[concept.split(":", 1)[1]][layer]
        else:
            return None
        return v / (np.linalg.norm(v) + 1e-8)

    @torch.no_grad()
    def generate(self, req: GenRequest) -> dict:
        tok, model = self.tok, self.model
        prompt = tok.apply_chat_template(req.messages, add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        plen = enc["input_ids"].shape[1]

        handle = None
        if req.steer.mode == "addition" and req.steer.concept not in ("none", ""):
            unit = self._unit(req.steer.concept, STEER_LAYER)
            coeff = req.steer.coefficient
            if unit is not None and coeff:
                u = torch.as_tensor(unit, dtype=torch.float32)

                def hook(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    uu = u.to(h.dtype).to(h.device)
                    h = h + coeff * h.norm(dim=-1, keepdim=True) * uu
                    return (h,) + out[1:] if isinstance(out, tuple) else h

                handle = self.layers[STEER_LAYER].register_forward_hook(hook)
        try:
            gen_kwargs: dict[str, Any] = dict(max_new_tokens=req.max_new_tokens,
                                              pad_token_id=tok.pad_token_id)
            if req.temperature and req.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=req.temperature, top_p=0.95)
            else:
                gen_kwargs.update(do_sample=False)
            out = model.generate(**enc, **gen_kwargs)
        finally:
            if handle is not None:
                handle.remove()

        reply_ids = out[0, plen:].tolist()
        res = model(out, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states[1:][MONITOR_LAYER][0]  # (seq, hidden)
        reply_hs = hs[plen:].float().cpu().numpy()

        # per token: deviation from the neutral-reply baseline, projected onto the
        # emotion directions (dominant = argmax). Special tokens (turn tags, eos)
        # are skipped so format markers don't leak into the display.
        toks = []
        raw_scores = []
        for j, tid in enumerate(reply_ids):
            if tid in self.special_ids:
                continue
            centered = reply_hs[j] - self.reply_baseline
            scores = self.emoU @ centered
            k = int(scores.argmax())
            toks.append({"text": tok.decode([tid]), "emotion": self.emo_names[k],
                         "score": float(scores[k])})
            raw_scores.append(float(scores[k]))
        s = np.array(raw_scores) if raw_scores else np.zeros(1)
        lo, hi = float(s.min()), float(s.max())
        for t in toks:
            norm = (t["score"] - lo) / (hi - lo + 1e-8)
            fam, color = self.emo_colors.get(t["emotion"], ("other", DEFAULT_COLOR))
            t["family"] = fam
            t["color"] = _hex_rgba(color, 0.10 + 0.80 * norm)

        mean_dev = reply_hs.mean(axis=0) - self.reply_baseline
        order = np.argsort(-(self.emoU @ mean_dev))[:6]
        top = [(self.emo_names[i], float((self.emoU @ mean_dev)[i])) for i in order]
        axis_score = float(reply_hs.mean(axis=0) @ self.axis.unit(MONITOR_LAYER))
        reply_text = tok.decode(out[0, plen:], skip_special_tokens=True).strip()
        return {"reply": reply_text, "tokens": toks,
                "emotions": [{"name": n, "score": round(v, 2)} for n, v in top],
                "axis_score": round(axis_score, 2)}


def build_app(scouter: Scouter) -> FastAPI:
    app = FastAPI(title="scouter-ui")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/concepts")
    def concepts() -> dict:
        return scouter.concepts()

    @app.get("/families")
    def families() -> dict:
        return {f: c for f, (c, _) in FAMILIES.items()}

    @app.post("/generate")
    def generate(req: GenRequest) -> dict:
        return scouter.generate(req)

    return app


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>scouter</title>
<style>
 :root{--bg:#0d0f1a;--panel:#161a2b;--ink:#e8ebf5;--muted:#8a90a6;--line:#2a3050;--accent:#3b6fe0}
 *{box-sizing:border-box} body{margin:0;font:14px/1.55 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
 header{padding:11px 18px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
 header h1{font-size:16px;margin:0;font-weight:600} header .sub{color:var(--muted);font-size:12px}
 .wrap{display:flex;height:calc(100vh - 49px)}
 .chat{flex:1;display:flex;flex-direction:column;min-width:0}
 #log{flex:1;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:10px}
 .msg{max-width:780px}
 .msg.user{align-self:flex-end;background:#21305a;padding:8px 12px;border-radius:10px}
 .msg.bot{align-self:flex-start;background:#11141f;border:1px solid var(--line);border-radius:10px;padding:8px 12px}
 .bot .meta{color:var(--muted);font-size:11px;margin-bottom:5px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 .bot .label{color:#cdd6ff;font-weight:600;background:#243; padding:1px 8px;border-radius:10px;background:#22305a}
 .tok{white-space:pre-wrap;border-radius:3px;padding:1px 0}
 .sibs{display:flex;gap:5px;flex-wrap:wrap;margin:2px 0 4px}
 .sib{font-size:11px;padding:2px 9px;border-radius:11px;border:1px solid var(--line);background:#161a2b;
      color:var(--muted);cursor:pointer}
 .sib.active{background:var(--accent);color:#fff;border-color:var(--accent)}
 .btnrow{margin-top:6px;display:flex;gap:8px}
 .mini{font-size:11px;padding:3px 9px;border-radius:7px;border:1px solid var(--line);background:#161a2b;color:#aab;cursor:pointer}
 .mini:hover{border-color:var(--accent);color:#fff}
 .composer{padding:10px 18px;border-top:1px solid var(--line)}
 .branches{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;align-items:center}
 .branches .chip{font-size:11px;padding:2px 8px;border-radius:11px;background:#22305a;color:#cdd6ff;cursor:pointer}
 .branches .chip b{margin-left:6px;opacity:.6}
 .crow{display:flex;gap:8px;align-items:center}
 .crow input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--ink);
   padding:10px 12px;border-radius:8px;font-size:14px}
 button.send{background:var(--accent);border:0;color:#fff;padding:10px 18px;border-radius:8px;cursor:pointer}
 .side{width:300px;border-left:1px solid var(--line);padding:16px;overflow:auto;background:#10131f}
 .side h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 8px}
 .side h3:first-child{margin-top:0}
 select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px;width:100%}
 .row{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin:8px 0 3px}
 input[type=range]{width:100%}
 .legend{display:flex;flex-wrap:wrap;gap:6px} .legend span{font-size:11px;padding:2px 7px;border-radius:10px;color:#0d0f1a;font-weight:600}
 .bars div{display:flex;align-items:center;gap:6px;font-size:12px;margin:3px 0}
 .bars .bar{height:8px;background:var(--accent);border-radius:4px}
 .addrow{display:flex;gap:6px} .addrow select{flex:1}
 .addrow button{background:#243a6e;border:0;color:#fff;border-radius:6px;padding:0 12px;cursor:pointer}
</style></head><body>
<header><h1>scouter</h1><span class="sub">gemma-4-12B · per-token emotion · branching steer tree</span></header>
<div class="wrap">
 <div class="chat">
   <div id="log"></div>
   <div class="composer">
     <div class="branches" id="branches"></div>
     <div class="crow">
       <input id="inp" type="text" placeholder="Message — fans out one branch per behavior below…" autocomplete="off">
       <button class="send" id="send">Send</button>
     </div>
   </div>
 </div>
 <div class="side">
   <h3>Branch behaviors</h3>
   <div class="addrow"><select id="concept"></select><button id="add">+</button></div>
   <div class="row"><span>mode</span></div>
   <select id="mode"><option value="addition">addition</option><option value="off">off (monitor only)</option></select>
   <div class="row"><span>coefficient</span><span id="coefval">0.22</span></div>
   <input id="coef" type="range" min="0" max="0.4" step="0.01" value="0.22">
   <div class="row"><span>temperature</span><span id="tempval">0.0</span></div>
   <input id="temp" type="range" min="0" max="1.2" step="0.05" value="0">
   <h3>Selected node</h3>
   <div id="axis" style="font-size:13px">assistant-axis: –</div>
   <div class="bars" id="emos"></div>
   <h3>Emotion colors</h3>
   <div class="legend" id="legend"></div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const log=$('log'), inp=$('inp');
let nodes={}, idc=0, active=null, branchList=[];   // branchList: array of concept strings

function node(role, data){const id=++idc; nodes[id]={id,role,parent:active,children:[],...data};
  if(active) active.children.push(nodes[id]); return nodes[id];}
function pathTo(n){const p=[];let c=n;while(c){p.unshift(c);c=c.parent;}return p;}
function messagesFor(userNode){return pathTo(userNode).filter(n=>n.role!=='root')
   .map(n=>({role:n.role==='user'?'user':'assistant',content:n.role==='user'?n.content:n.reply}));}

$('coef').oninput=e=>$('coefval').textContent=(+e.target.value).toFixed(2);
$('temp').oninput=e=>$('tempval').textContent=(+e.target.value).toFixed(2);

async function init(){
  const c=await (await fetch('/concepts')).json();
  const sel=$('concept');
  const add=(l,v)=>{const o=document.createElement('option');o.textContent=l;o.value=v;sel.appendChild(o);};
  add('none (no steering)','none'); add('assistant axis','assistant_axis');
  const g1=document.createElement('optgroup');g1.label='personas';
  c.personas.forEach(p=>{const o=document.createElement('option');o.textContent='persona: '+p;o.value='persona:'+p;g1.appendChild(o);});sel.appendChild(g1);
  const g2=document.createElement('optgroup');g2.label='emotions';
  c.emotions.forEach(p=>{const o=document.createElement('option');o.textContent='emotion: '+p;o.value='emotion:'+p;g2.appendChild(o);});sel.appendChild(g2);
  const fam=await (await fetch('/families')).json();
  Object.entries(fam).forEach(([f,col])=>{const s=document.createElement('span');s.textContent=f;s.style.background=col;$('legend').appendChild(s);});
  branchList=['none']; renderBranches();
  // virtual root
  active={id:0,role:'root',children:[],parent:null}; nodes[0]=active;
}
function renderBranches(){
  const b=$('branches');b.innerHTML='<span style="color:#8a90a6;font-size:11px">branches:</span>';
  branchList.forEach((c,i)=>{const s=document.createElement('span');s.className='chip';
    s.innerHTML=c+'<b>✕</b>';s.onclick=()=>{branchList.splice(i,1);if(!branchList.length)branchList=['none'];renderBranches();};
    b.appendChild(s);});
}
$('add').onclick=()=>{const v=$('concept').value;if(!branchList.includes(v)){if(branchList.length===1&&branchList[0]==='none')branchList=[];branchList.push(v);renderBranches();}};

function label(steer){return (steer.mode==='off'||steer.concept==='none')?'none':steer.concept;}

async function genBranch(userNode, steer){
  const body={messages:messagesFor(userNode),steer,temperature:+$('temp').value,max_new_tokens:90};
  const r=await (await fetch('/generate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})).json();
  const prev=active; active=userNode;
  const bot=node('bot',{reply:r.reply,tokens:r.tokens,emotions:r.emotions,axis_score:r.axis_score,label:label(steer),steer});
  active=prev; return bot;
}
async function send(){
  const text=inp.value.trim(); if(!text) return; inp.value='';
  const u=node('user',{content:text});       // child of current active
  const mode=$('mode').value, coef=+$('coef').value;
  active=u;
  const branches=[];
  for(const concept of branchList){
    const steer={concept,mode:concept==='none'?'off':mode,coefficient:coef};
    branches.push(await genBranch(u,steer));
  }
  active=branches[0]; render();
}
async function retry(bot){
  const u=bot.parent; const n=await genBranch(u,bot.steer); active=n; render();
}
function setActive(n){active=n; render(); updateSide(n);}

function botEl(n){
  const d=document.createElement('div');d.className='msg bot';
  const meta=document.createElement('div');meta.className='meta';
  meta.innerHTML='<span class="label">'+n.label+'</span><span>axis '+n.axis_score+'</span><span>top: '+
     n.emotions.slice(0,3).map(e=>e.name).join(', ')+'</span>';
  d.appendChild(meta);
  const body=document.createElement('div');
  n.tokens.forEach(t=>{const s=document.createElement('span');s.className='tok';s.textContent=t.text;
     s.style.background=t.color;s.title=t.emotion+' ('+t.score.toFixed(1)+')';body.appendChild(s);});
  d.appendChild(body);
  const br=document.createElement('div');br.className='btnrow';
  const rt=document.createElement('button');rt.className='mini';rt.textContent='↻ retry';rt.onclick=()=>retry(n);
  const cf=document.createElement('button');cf.className='mini';cf.textContent='⌖ continue from here';cf.onclick=()=>setActive(n);
  br.appendChild(rt);br.appendChild(cf);d.appendChild(br);
  return d;
}
function sibChips(n){
  if(!n.parent||n.parent.children.length<2)return null;
  const w=document.createElement('div');w.className='sibs';
  n.parent.children.forEach(s=>{const c=document.createElement('span');
    c.className='sib'+(s===n?' active':'');c.textContent=s.role==='bot'?s.label:('“'+s.content.slice(0,18)+'…”');
    c.onclick=()=>setActive(s);w.appendChild(c);});
  return w;
}
function render(){
  log.innerHTML='';
  pathTo(active).filter(n=>n.role!=='root').forEach(n=>{
    const sc=sibChips(n); if(sc)log.appendChild(sc);
    if(n.role==='user'){const d=document.createElement('div');d.className='msg user';d.textContent=n.content;log.appendChild(d);}
    else log.appendChild(botEl(n));
  });
  log.scrollTop=log.scrollHeight;
  const last=pathTo(active).filter(n=>n.role==='bot').pop(); if(last)updateSide(last);
}
function updateSide(n){
  $('axis').textContent='assistant-axis: '+n.axis_score+'   ·   behavior: '+n.label;
  const e=$('emos');e.innerHTML='';const mx=Math.max(...n.emotions.map(x=>Math.abs(x.score)),1);
  n.emotions.forEach(x=>{const r=document.createElement('div');
    r.innerHTML='<span style="width:96px">'+x.name+'</span><div class="bar" style="width:'+(80*Math.abs(x.score)/mx)+'px"></div><span>'+x.score+'</span>';
    e.appendChild(r);});
}
$('send').onclick=send; inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
init();
</script></body></html>"""



def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # for steer_matrix import
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-4-12b-it")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    scouter = Scouter(args.model)
    print(f"loading {args.model} (4-bit) ...", flush=True)
    scouter.load()
    print(f"ready -> http://{args.host}:{args.port}", flush=True)
    uvicorn.run(build_app(scouter), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
