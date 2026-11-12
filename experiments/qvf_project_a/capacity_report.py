"""Interactive report for longer training, fixed-qubit capacity and spatial design."""

import argparse
import base64
import html
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Longer training and larger QVF capacity</title><style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground,#ddd);color-scheme:light dark}*{box-sizing:border-box}body{margin:0}p{line-height:1.55;margin:10px 0 14px}h2{font-size:20px;margin:0 0 12px}
.panel{padding:16px;background:var(--card,#222);border:1px solid var(--border,#444);border-radius:var(--radius,12px);margin:16px 0}.muted,figcaption{font-size:13px;line-height:1.5;color:var(--muted-foreground,#aaa)}
label{display:block;margin:12px 0 6px;font-size:14px}select{width:100%;padding:8px;font:inherit;border:1px solid var(--border,#555);border-radius:6px;background:var(--background,#222);color:var(--foreground,#ddd)}input{width:100%;accent-color:var(--primary,#8ab4ff)}
.pictures{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}canvas{display:block;width:100%;height:auto;image-rendering:pixelated;border-radius:4px}figcaption{margin-top:7px}.chart{height:230px;width:100%}.chart svg{display:block;width:100%;height:100%}.stats{font-variant-numeric:tabular-nums}.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;line-height:1.5}.flow span{padding:7px 9px;border:1px solid var(--border,#555);border-radius:6px}.flow b{font-weight:400}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:13px}.legend span:before{content:'';display:inline-block;width:10px;height:3px;background:var(--chart-1,#69a7ff);margin-right:5px;vertical-align:middle}.legend span:nth-child(2):before{background:var(--chart-2,#f3b562)}a{color:var(--primary,#8ab4ff)}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px 4px;text-align:right;border-bottom:1px solid var(--border,#444)}td:first-child,th:first-child{text-align:left}
@media(max-width:480px){h2{font-size:18px}.pictures{gap:7px;grid-template-columns:repeat(2,minmax(0,1fr))}figcaption{font-size:12px}}
</style></head><body>__ASSETS__
<p><strong>Longer training helped modestly. Widening the decoder improved its training fit, but did not improve held-out PSNR.</strong> All six requested runs reached 50,000 updates. Both variants use the same VAE encoder, 64-number code, 4,744 training images, 256 held-out images, five qubits, circuit depth five and original RGB readout. The first experiment changes training duration; the second changes only the amplitude network's width at that longer budget.</p>
<div class="panel"><h2>1. What changed reconstruction accuracy?</h2><div id="summary-chart" class="chart" role="img" aria-label="Held-out reconstruction accuracy for the earlier run, longer training and larger amplitude network"></div><p id="summary" class="stats"></p><p id="changes"></p>
<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Updates</th><th>Decoder weights</th><th>Qubits</th></tr></thead><tbody><tr><td>Previous width 128</td><td>7,680</td><td>48,950</td><td>5</td></tr><tr><td>Width 128, longer</td><td>50,000</td><td>48,950</td><td>5</td></tr><tr><td>Width 512, longer</td><td>50,000</td><td>588,470</td><td>5</td></tr></tbody></table></div>
<p class="muted">Dots show three-seed means, whiskers one sample standard deviation. Horizontal labels show updates / hidden width. These are development validation measurements, not final test-set results. Higher PSNR means lower squared pixel error; inspect the images below to judge detail. The wider decoder has about 12 times as many parameters and uses more computation per update. Its quantum circuit is unchanged.</p></div>
<div class="panel"><h2>2. Inspect the reconstructions</h2><label for="split">Which images to inspect</label><select id="split"><option value="validation">Held-out images: all 256</option><option value="train">Training images: first 64 in the fixed split</option></select><label for="seed">Training seed</label><select id="seed"><option value="0">Seed 0</option><option value="1">Seed 1</option><option value="2">Seed 2</option></select>
<label for="stage">Which checkpoint to compare</label><select id="stage"><option value="selected">Selected by validation objective</option><option value="final">Final checkpoint at the stated update budget</option></select>
<label for="photo">Photograph <output id="photo-number">1</output> of <span id="photo-count">256</span></label><input id="photo" type="range" min="0" max="255" step="1" value="0">
<div class="pictures"><figure><canvas id="original" width="32" height="32" role="img" aria-label="Original image"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="previous" width="32" height="32" role="img" aria-label="Previous short-training reconstruction"></canvas><figcaption>Width 128<br>7,680 updates</figcaption></figure><figure><canvas id="baseline" width="32" height="32" role="img" aria-label="Longer-training vanilla QVF reconstruction"></canvas><figcaption>Width 128<br>50,000 updates</figcaption></figure><figure><canvas id="wide" width="32" height="32" role="img" aria-label="Wider QVF reconstruction"></canvas><figcaption>Width 512<br>50,000 updates</figcaption></figure></div>
<p id="image-stats" class="muted stats"></p><p id="checkpoint-steps" class="muted stats"></p><p class="muted">All images use encoder mean codes, without per-image fitting. Original photographs are 32 × 32 pixels and displayed without smoothing. Raw predictions are scored before RGB clipping. Changing the checkpoint selector also updates the accuracy chart.</p></div>
<div class="panel"><h2>3. Did learning continue beyond the old stopping point?</h2><div id="learning" class="chart" role="img" aria-label="Held-out learning curves for original and wider QVF at 50000 updates"></div><div class="legend"><span>Width 128</span><span>Width 512</span></div><p class="muted">This follows the training seed selected above and shows steps 2,560 through 50,000. The vertical marker is the old 7,680-update endpoint. Both use the same constant learning rate; no new scheduler was introduced.</p>
<p id="fit-summary" class="stats"></p><p class="muted">Training scores use all 4,744 images. A gap between training and validation accuracy can reveal overfitting. Checkpoint selection minimizes sampled-code MSE plus the existing KL penalty, so the selected step need not maximize mean-code PSNR.</p></div>
<p>The larger model learns the training images more closely, while its accuracy on unseen images deteriorates with further training. This is evidence of overfitting under our current recipe. Extra training helps the smaller model, but extra pointwise width alone does not solve the generalization problem. These runs do not establish that we have reached the best possible learning schedule or a fundamental five-qubit limit.</p>
<div class="panel"><h2>4. A way to make QVF more like an image decoder</h2><p>A conventional convolutional decoder builds spatial features and reuses local filters across the image. Our current amplitude MLP processes each coordinate separately, with spatial relationships represented indirectly through the shared image code and weights.</p>
<div class="flow" aria-label="Proposed local-feature QVF decoder"><span>Same 64-number code</span><b>→</b><span>Convolutional feature grid</span><b>→</b><span>Local feature + coordinate</span><b>→</b><span>32 amplitudes</span><b>→</b><span>Same 5-qubit circuit</span><b>→</b><span>RGB</span></div>
<p>The prototype has 299,894 decoder parameters. It generates a 16 × 16 grid of 32-channel features from the global code, then interpolates local features for each QVF query. It receives no extra image information and keeps the original quantum readout.</p>
<p class="muted"><strong>This is an untrained prototype.</strong> Interpolation, gradient, query-chunking and GPU checks pass. It is not included in the measured width comparison. Local features plus coordinate queries have a classical precedent in <a href="https://arxiv.org/abs/2012.09161" target="_blank" rel="noopener">LIIF</a>; this proposal generates its features from the VAE code and retains QVF's measurement head. The <a href="https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py" target="_blank" rel="noopener">latent-diffusion decoder</a> provides a reference for convolution and upsampling.</p>
<p class="muted">A future width-357 pointwise QVF would have 299,705 parameters, allowing a close capacity match when testing whether spatial structure helps. A spatial feature grid generated from 64 numbers is different from enlarging the VAE's latent code into a grid.</p></div>
<p id="verification" class="muted"></p>
<script>
const data=__DATA__,$=id=>document.getElementById(id),labels=['previous','baseline50k','wide51250k'],short=['7,680 / 128','50,000 / 128','50,000 / 512'],fmt=x=>x.toFixed(2),signed=x=>(x>=0?'+':'')+x.toFixed(2),seed=$('seed'),stage=$('stage'),photo=$('photo'),split=$('split');
$('verification').textContent=`All ${data.verification.run_count} new runs reached 50,000 updates and passed raw-metric, split, source-snapshot and checkpoint checks. Every baseline validation record through step 7,680 exactly reproduces the earlier run. Previous checkpoints and the original simulator are unchanged. No diffusion model was trained.`;
function metric(){return stage.value==='selected'?'selected_validation_psnr':'final_validation_psnr';}
function axes(el,min,max){const w=el.clientWidth,h=230,L=43,R=18,T=24,B=45,sy=v=>h-B-(v-min)/(max-min)*(h-T-B);let parts='';for(let i=0;i<=4;i++){const v=min+(max-min)*i/4;parts+=`<line x1="${L}" x2="${w-R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--border,#555)"/><text x="${L-7}" y="${sy(v)+4}" text-anchor="end" fill="var(--foreground,#ddd)" font-size="11">${v.toFixed(1)}</text>`;}return{w,h,L,R,T,B,sy,parts};}
function svg(el,a,extra){el.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${a.w} ${a.h}">${a.parts}${extra}</svg>`;}
function summary(){const key=metric(),values=labels.map(l=>data.summaries[l][key]),el=$('summary-chart'),a=axes(el,Math.floor(Math.min(...values.map(r=>r.mean-r.sample_sd))-.25),Math.ceil(Math.max(...values.map(r=>r.mean+r.sample_sd))+.25));let p='';values.forEach((r,i)=>{const x=a.L+(i+.5)/3*(a.w-a.L-a.R),lo=a.sy(r.mean-r.sample_sd),hi=a.sy(r.mean+r.sample_sd),c=`var(--chart-${i+1},#69a7ff)`;p+=`<path d="M${x},${lo}V${hi}M${x-5},${lo}h10M${x-5},${hi}h10" stroke="${c}" stroke-width="2"/><circle cx="${x}" cy="${a.sy(r.mean)}" r="5" fill="${c}"/><text x="${x}" y="${hi-9}" text-anchor="middle" fill="var(--foreground,#ddd)" font-size="12">${fmt(r.mean)}</text><text x="${x}" y="${a.h-19}" text-anchor="middle" fill="var(--foreground,#ddd)" font-size="10">${short[i]}</text>`;});svg(el,a,p);$('summary').textContent=`${stage.value==='selected'?'Validation-selected':'Final'} checkpoints: earlier ${fmt(values[0].mean)} ± ${fmt(values[0].sample_sd)} dB; longer training ${fmt(values[1].mean)} ± ${fmt(values[1].sample_sd)} dB; wider field ${fmt(values[2].mean)} ± ${fmt(values[2].sample_sd)} dB.`;$('changes').textContent=`Mean paired change from longer training: ${signed(data.comparisons.longer_training[key].mean)} dB. Additional change from widening the field at 50,000 updates: ${signed(data.comparisons.wider_field[key].mean)} dB.`;}
function crop(id,img,i,group,rows){const ctx=$(id).getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(img,1+(i%16)*33,1+(Math.floor(i/16)+group*rows)*33,32,32,0,0,32,32)}
function pictures(){const s=+seed.value,asset=data.images.find(r=>r.seed===s&&r.stage===stage.value&&r.group===split.value);photo.max=asset.count-1;const i=Math.min(+photo.value,asset.count-1);photo.value=i;['original','previous','baseline','wide'].forEach((id,j)=>crop(id,asset.asset,i,j,asset.group_rows));$('photo-number').textContent=i+1;$('photo-count').textContent=asset.count;const key=stage.value+(split.value==='train'?'_train':'')+'_per_image_psnr',runs=labels.map(l=>data.groups[l].find(r=>r.seed===s)),values=runs.map(r=>r[key][i]);$('image-stats').textContent=`Dataset image ${asset.ids[i]}. Earlier: ${fmt(values[0])} dB; longer: ${fmt(values[1])} dB; wider: ${fmt(values[2])} dB.`;const steps=runs.map(r=>(stage.value==='selected'?r.selected_step:r.steps).toLocaleString());$('checkpoint-steps').textContent=`Actual checkpoint steps: earlier ${steps[0]}; longer ${steps[1]}; wider ${steps[2]}.`;}
function learning(){const s=+seed.value,runs=['baseline50k','wide51250k'].map(l=>data.groups[l].find(r=>r.seed===s)),curves=runs.map(r=>r.curve.filter(p=>p[0]>=2560)),el=$('learning'),a=axes(el,Math.floor(Math.min(...curves.flat().map(p=>p[1]))-.2),Math.ceil(Math.max(...curves.flat().map(p=>p[1]))+.2)),sx=v=>a.L+(v-2560)/(50000-2560)*(a.w-a.L-a.R);let p=`<line x1="${sx(7680)}" x2="${sx(7680)}" y1="${a.T}" y2="${a.h-a.B}" stroke="var(--muted-foreground,#aaa)" stroke-dasharray="4 3"/>`;curves.forEach((curve,j)=>p+=`<path d="${curve.map((r,i)=>`${i?'L':'M'}${sx(r[0])},${a.sy(r[1])}`).join(' ')}" fill="none" stroke="var(--chart-${j+1},#69a7ff)" stroke-width="2"/>`);[7680,25000,50000].forEach(v=>p+=`<text x="${sx(v)}" y="${a.h-19}" text-anchor="middle" fill="var(--foreground,#ddd)" font-size="11">${v.toLocaleString()}</text>`);svg(el,a,p);$('fit-summary').textContent=`Seed ${s}, final checkpoints: width 128 training ${fmt(runs[0].final_train_psnr)} dB / held-out ${fmt(runs[0].final_validation_psnr)} dB; width 512 training ${fmt(runs[1].final_train_psnr)} dB / held-out ${fmt(runs[1].final_validation_psnr)} dB.`;}
data.images.forEach((r,i)=>{r.asset=$(`asset-${i}`)});
summary();learning();
function activate(){if(window.qvfReportReady)return;seed.addEventListener('change',()=>{pictures();learning()});stage.addEventListener('change',()=>{pictures();summary()});split.addEventListener('change',pictures);photo.addEventListener('input',pictures);pictures();window.qvfReportReady=true;let width=0;new ResizeObserver(()=>{const next=document.body.clientWidth;if(next!==width){width=next;summary();learning()}}).observe(document.body)}
if(data.images.every(r=>r.asset.complete&&r.asset.naturalWidth>0)){activate()}else{window.addEventListener('load',activate,{once:true})}
</script></body></html>'''


def build(data, local=False):
    assets = []
    for i, item in enumerate(data["images"]):
        path = Path(item["image"])
        # Optional lossless display copies preserve every RGB pixel while
        # reducing the size of the portable and inline reports.
        if path.with_suffix(".webp").exists():
            path = path.with_suffix(".webp")
        mime = "image/webp" if path.suffix == ".webp" else "image/png"
        source = str(path) if local else f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        assets.append(f'<img id="asset-{i}" src="{html.escape(source, quote=True)}" alt="">')
    # Local-image inlining also sees paths inside JSON. Keep each atlas only in
    # its img element so the inline viewer does not embed every image twice.
    report_data = dict(data)
    report_data["images"] = [{key: value for key, value in item.items() if key != "image"}
                             for item in data["images"]]
    return TEMPLATE.replace("__ASSETS__", '<div hidden>' + ''.join(assets) + '</div>').replace(
        "__DATA__", json.dumps(report_data, allow_nan=False).replace("</", "<\\/"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    outputs = [(Path(str(args.output_stem) + ".html"), False), (Path(str(args.output_stem) + "_local.html"), True)]
    if any(path.exists() for path, _ in outputs):
        raise FileExistsError("Choose a fresh report output stem")
    data = json.loads(args.data.read_text())
    for path, local in outputs:
        with path.open("x") as handle:
            handle.write(build(data, local))
        print(path.resolve())


if __name__ == "__main__":
    main()
