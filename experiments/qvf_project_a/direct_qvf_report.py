"""Interactive, training-only comparison of direct QVF fitting and the QVF VAE."""

import argparse
import base64
import html
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QVF alone on the larger image set</title><style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground);color-scheme:light dark}*{box-sizing:border-box}body{margin:0}p{line-height:1.55;margin:10px 0 14px}h2{font-size:20px;margin:0 0 12px}.panel{padding:16px;border:1px solid var(--border);border-radius:var(--radius);background:var(--card);margin:16px 0}.muted,figcaption{font-size:13px;line-height:1.5;color:var(--muted-foreground)}.chart{height:220px;width:100%}.chart svg{width:100%;height:100%;display:block}.pictures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}canvas{display:block;width:100%;height:auto;image-rendering:pixelated;border-radius:4px}figcaption{margin-top:7px}label{display:block;margin:12px 0 6px;font-size:14px}select{width:100%;padding:8px;font:inherit;color:var(--foreground);background:var(--background);border:1px solid var(--border);border-radius:6px}input{width:100%;accent-color:var(--primary)}.stats{font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:8px 4px;border-bottom:1px solid var(--border)}td:first-child,th:first-child{text-align:left}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:13px}.legend span:before{content:'';display:inline-block;width:13px;height:3px;background:var(--chart-2);margin-right:5px;vertical-align:middle}.legend span:nth-child(2):before{background:var(--chart-1)}@media(max-width:480px){h2{font-size:18px}.pictures{gap:6px}figcaption{font-size:11px}}
</style></head><body>__ASSETS__
<p><strong>We tested the original shared QVF decoder on all 4,744 training images, without a VAE encoder.</strong> It learns one 64-number code per image directly from that image's pixels. The decoder keeps width 128, five qubits, circuit depth five and the original RGB readout. All three seeds completed 50,000 updates.</p>
<div class="panel"><h2>How well can each system fit the same images?</h2><div id="summary-chart" class="chart" role="img" aria-label="Full training set PSNR, QVF VAE versus direct QVF fitting"></div><p id="summary" class="stats"></p><p id="edge" class="muted stats"></p>
<p class="muted">Both scores use all 1,024 pixels of all 4,744 fitted training images and the final 50,000-update checkpoint. Dots show the mean over three seeds; whiskers show one sample standard deviation. This measures fitting ability. It does not measure reconstruction of unseen images or image generation.</p></div>
<div class="panel"><h2>Check the detail on matching images</h2><label for="seed">Training seed</label><select id="seed"><option value="0">Seed 0</option><option value="1">Seed 1</option><option value="2">Seed 2</option></select><label for="photo">Training image <output id="photo-number">1</output> of 256 in the fixed gallery</label><input id="photo" type="range" min="0" max="255" value="0" step="1">
<div class="pictures"><figure><canvas id="original" width="32" height="32" role="img" aria-label="Original training image"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="vae" width="32" height="32" role="img" aria-label="VAE reconstruction using the encoder mean code"></canvas><figcaption>QVF VAE<br>Encoder mean code</figcaption></figure><figure><canvas id="direct" width="32" height="32" role="img" aria-label="Direct QVF reconstruction using the fitted image code"></canvas><figcaption>QVF alone<br>Directly fitted code</figcaption></figure></div>
<p id="image-stats" class="muted stats"></p><p class="muted">Gallery positions were fixed before training, spaced across the training list. All 4,744 raw reconstructions are saved. Images are displayed at their native 32 × 32 resolution without smoothing; clipping is only for display. The scores use unclipped predictions.</p></div>
<div class="panel"><h2>Was direct fitting still learning?</h2><div id="learning" class="chart" role="img" aria-label="Direct QVF full-training PSNR over optimizer updates"></div><div class="legend"><span>Direct QVF fitting curve</span><span>VAE final training score, reference only</span></div><p id="curve-note" class="muted stats"></p><p class="muted">The curve follows the seed selected above, starting at 5,000 updates. Its points evaluate the full training set. The horizontal line is the VAE's final training score, not its learning curve. Each image contributes gradients through its own code about 337 times during the 50,000 shared-field updates.</p></div>
<div class="panel"><h2>What this test separates</h2><p>QVF alone can adjust every image's code independently. The VAE must learn an encoder that supplies codes and must also satisfy its sampling and KL objective. If direct fitting recovers more detail, the same QVF field can represent more than the VAE currently reconstructs.</p>
<p>This comparison changes code inference and regularization together. It cannot tell us how much comes from the encoder, the VAE penalty, or optimization. Persistent blur would also leave optimization and finite capacity as possibilities; it would not establish a fundamental five-qubit limit.</p>
<div class="table-wrap"><table><thead><tr><th>Setting</th><th>QVF VAE</th><th>QVF alone</th></tr></thead><tbody><tr><td>Shared decoder parameters</td><td>48,950</td><td>48,950</td></tr><tr><td>How codes are obtained</td><td>Encoder</td><td>Per-image optimization</td></tr><tr><td>Encoder parameters</td><td>427,872</td><td>0</td></tr><tr><td>Free image-code entries</td><td>0</td><td>303,616</td></tr><tr><td>Regularization</td><td>VAE KL</td><td>Original code-norm penalty</td></tr><tr><td>Gradient clipping</td><td>Norm 1</td><td>None, original convention</td></tr></tbody></table></div>
<p class="muted">Both use constant Adam learning rates of 0.001, batches of 32 images and 256 sampled pixels per image per update. QVF alone uses separate Adam optimizers for its field and code table. Its code penalty is 0.0001 times the mean code norm. This is the unchanged decoder with a larger-data fitting protocol, not a reproduction of a published large-data benchmark. No new convolutions or diffusion were trained.</p></div>
<p id="verification" class="muted"></p>
<script>
const data=__DATA__,$=id=>document.getElementById(id),seed=$('seed'),photo=$('photo'),fmt=x=>x.toFixed(2);
function axes(el,min,max){const w=el.clientWidth,h=220,L=43,R=18,T=24,B=40,sy=v=>h-B-(v-min)/(max-min)*(h-T-B);let p='';for(let i=0;i<=4;i++){const v=min+(max-min)*i/4;p+=`<line x1="${L}" x2="${w-R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--border)"/><text x="${L-7}" y="${sy(v)+4}" text-anchor="end" fill="var(--foreground)" font-size="11">${v.toFixed(1)}</text>`}return{w,h,L,R,T,B,sy,p}}
function show(el,a,p){el.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${a.w} ${a.h}">${a.p}${p}</svg>`}
function summary(){const vals=['vae','direct'].map(k=>data.summary[k].psnr_db),el=$('summary-chart'),a=axes(el,Math.floor(Math.min(...vals.map(r=>r.mean-r.sample_sd))-.3),Math.ceil(Math.max(...vals.map(r=>r.mean+r.sample_sd))+.3));let p='';vals.forEach((r,i)=>{const x=a.L+(i+.5)/2*(a.w-a.L-a.R),lo=a.sy(r.mean-r.sample_sd),hi=a.sy(r.mean+r.sample_sd),c=`var(--chart-${i+1})`;p+=`<path d="M${x},${lo}V${hi}M${x-5},${lo}h10M${x-5},${hi}h10" stroke="${c}" stroke-width="2"/><circle cx="${x}" cy="${a.sy(r.mean)}" r="5" fill="${c}"/><text x="${x}" y="${hi-9}" text-anchor="middle" fill="var(--foreground)" font-size="12">${fmt(r.mean)} dB</text><text x="${x}" y="${a.h-14}" text-anchor="middle" fill="var(--foreground)" font-size="12">${i?'QVF alone':'QVF VAE'}</text>`});show(el,a,p);const delta=data.paired_psnr_change.mean;$('summary').textContent=`Training PSNR: QVF VAE ${fmt(vals[0].mean)} ± ${fmt(vals[0].sample_sd)} dB; QVF alone ${fmt(vals[1].mean)} ± ${fmt(vals[1].sample_sd)} dB. Paired change: ${delta>=0?'+':''}${fmt(delta)} dB.`;$('edge').textContent=`Adjacent-pixel error, lower is better: VAE ${data.summary.vae.gradient_mse.mean.toFixed(6)}; direct QVF ${data.summary.direct.gradient_mse.mean.toFixed(6)}. This checks local pixel differences; PSNR and edge error are not complete measures of perceived sharpness.`}
function pictures(){const s=+seed.value,i=+photo.value,asset=data.images.find(r=>r.seed===s),run=data.runs.find(r=>r.seed===s);['original','vae','direct'].forEach((id,j)=>{const ctx=$(id).getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(asset.asset,1+(i%16)*33,1+(Math.floor(i/16)+j*16)*33,32,32,0,0,32,32)});$('photo-number').textContent=i+1;$('image-stats').textContent=`Training-list position ${asset.training_positions[i]+1}; dataset image ${asset.dataset_indices[i]}. VAE: ${fmt(run.vae_gallery_psnr[i])} dB. QVF alone: ${fmt(run.direct_gallery_psnr[i])} dB.`}
function learning(){const run=data.runs.find(r=>r.seed===+seed.value),curve=run.curve.filter(r=>r[0]>=5000),el=$('learning'),vals=[run.vae.psnr_db,...curve.map(r=>r[1])],a=axes(el,Math.floor(Math.min(...vals)-.2),Math.ceil(Math.max(...vals)+.2)),sx=v=>a.L+(v-5000)/45000*(a.w-a.L-a.R);let p=`<line x1="${a.L}" x2="${a.w-a.R}" y1="${a.sy(run.vae.psnr_db)}" y2="${a.sy(run.vae.psnr_db)}" stroke="var(--chart-1)" stroke-dasharray="5 4"/><path d="${curve.map((r,i)=>`${i?'L':'M'}${sx(r[0])},${a.sy(r[1])}`).join(' ')}" fill="none" stroke="var(--chart-2)" stroke-width="2"/>`;[5000,25000,50000].forEach(v=>p+=`<text x="${sx(v)}" y="${a.h-14}" text-anchor="middle" fill="var(--foreground)" font-size="11">${v.toLocaleString()}</text>`);show(el,a,p);$('curve-note').textContent=`Seed ${run.seed}: final training PSNR ${fmt(run.direct.psnr_db)} dB; lowest recorded fitting MSE at step ${run.best_fitting_step.toLocaleString()}.`}
data.images.forEach((r,i)=>{r.asset=$(`asset-${i}`)});summary();learning();$('verification').textContent=`All ${data.verification.run_count} runs passed source, data-ID, budget, image-visit and raw-metric checks. Every training image was rendered again from both the direct-QVF and VAE checkpoints. Original QVF source and prior VAE checkpoints are unchanged. The image gallery is predetermined and its display compression is lossless.`;
function activate(){if(window.qvfDirectReady)return;seed.addEventListener('change',()=>{pictures();learning()});photo.addEventListener('input',pictures);pictures();window.qvfDirectReady=true;let width=0;new ResizeObserver(()=>{if(width!==document.body.clientWidth){width=document.body.clientWidth;summary();learning()}}).observe(document.body)}
if(data.images.every(r=>r.asset.complete&&r.asset.naturalWidth>0)){activate()}else{window.addEventListener('load',activate,{once:true})}
</script></body></html>'''


def build(data, local=False):
    assets = []
    for i, item in enumerate(data["images"]):
        path = Path(item["path"])
        source = str(path) if local else "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode()
        assets.append(f'<img id="asset-{i}" src="{html.escape(source, quote=True)}" alt="">')
    display = {key: value for key, value in data.items() if key != "protocol"}
    display["images"] = [{key: value for key, value in item.items() if key != "path"} for item in data["images"]]
    return TEMPLATE.replace("__ASSETS__", '<div hidden>' + ''.join(assets) + '</div>').replace(
        "__DATA__", json.dumps(display, allow_nan=False).replace("</", "<\\/"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(str(args.output_stem) + suffix) for suffix in (".html", "_local.html")]
    if any(p.exists() for p in paths):
        raise FileExistsError("Choose a fresh report output stem")
    data = json.loads(args.data.read_text())
    for path, local in zip(paths, (False, True)):
        with path.open("x") as handle:
            handle.write(build(data, local))
        print(path.resolve())


if __name__ == "__main__":
    main()
