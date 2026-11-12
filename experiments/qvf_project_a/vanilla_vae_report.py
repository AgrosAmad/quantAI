"""Show the verified VAE connection, fitting test and held-out reconstructions."""

import argparse
import base64
import html
import json
from pathlib import Path


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAE connected to vanilla QVF</title><style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground,#ddd);color-scheme:light dark}*{box-sizing:border-box}body{margin:0}p{line-height:1.55;margin:10px 0 14px}h2{font-size:20px;margin:0 0 12px}
.panel{padding:16px;background:var(--card,#222);border:1px solid var(--border,#444);border-radius:var(--radius,12px);margin:16px 0}.muted,figcaption{font-size:13px;line-height:1.5;color:var(--muted-foreground,#aaa)}
label{display:block;margin:12px 0 6px;font-size:14px}select{width:100%;padding:8px;font:inherit;border:1px solid var(--border,#555);border-radius:6px;background:var(--background,#222);color:var(--foreground,#ddd)}input{width:100%;accent-color:var(--primary,#8ab4ff)}
.pictures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.four{grid-template-columns:repeat(4,minmax(0,1fr))}figure{margin:0}canvas{display:block;width:100%;height:auto;image-rendering:pixelated;border-radius:4px}figcaption{margin-top:7px}.chart{height:225px;width:100%}.chart svg{display:block;width:100%;height:100%}.stats{font-variant-numeric:tabular-nums}.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;line-height:1.6}.flow span{padding:7px 9px;border:1px solid var(--border,#555);border-radius:6px}.flow b{font-weight:400}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:13px}.legend span:before{content:'';display:inline-block;width:10px;height:3px;background:var(--chart-1,#69a7ff);margin-right:5px;vertical-align:middle}.legend span:nth-child(2):before{background:var(--chart-2,#f3b562)}
@media(max-width:480px){h2{font-size:18px}.pictures{gap:7px}.four{grid-template-columns:repeat(2,minmax(0,1fr))}figcaption{font-size:12px}}
</style></head><body>__ASSETS__
<p><strong>The VAE now uses the verified vanilla QVF decoder.</strong> The image encoder supplies its 64-number code. The decoder is the same audited QVF model, including its amplitude network, quantum circuit and original RGB readout.</p>
<div class="flow" aria-label="Image to encoder to code to vanilla QVF to reconstruction"><span>Image</span><b>→</b><span>VAE encoder</span><b>→</b><span>64-number code</span><b>→</b><span>Vanilla QVF + pixel coordinates</span><b>→</b><span>Image</span></div>
<p class="muted">These experiments train the encoder and field together from scratch. The old optimized image-code table is replaced by encoder predictions. Mean codes give a repeatable reconstruction; sampled codes include the VAE's learned randomness. Diffusion has not been retrained.</p>
<div class="panel"><h2>1. Can the VAE reconstruct the same 50 images?</h2>
<p id="fit-summary"></p><label for="fit-photo">Training photograph <output id="fit-number">1</output> of 50</label><input id="fit-photo" type="range" min="0" max="49" step="1" value="0">
<label for="fit-code">Code supplied by the VAE</label><select id="fit-code"><option value="mean">Encoder mean</option><option value="sample">A posterior sample</option></select>
<div class="pictures"><figure><canvas id="fit-original" width="32" height="32" role="img" aria-label="Original training image"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="fit-direct" width="32" height="32" role="img" aria-label="Vanilla QVF with a directly optimized code"></canvas><figcaption>Vanilla QVF<br>Optimized image code</figcaption></figure><figure><canvas id="fit-vae" width="32" height="32" role="img" aria-label="Vanilla QVF with an encoder-produced code"></canvas><figcaption>Vanilla QVF<br>VAE encoder code</figcaption></figure></div>
<p id="fit-stats" class="muted stats"></p><p class="muted">The same 50 training images, 25,000 updates and all 1,024 pixels per image. The VAE uses our existing sampling, KL penalty and constant learning rate. The direct-fit baseline optimizes image codes with its own code penalty and learning-rate scheduler. This is a fitting check, not an isolated comparison of one training choice. Both panels use final checkpoints.</p><p id="fit-generalization" class="muted"></p></div>
<div class="panel"><h2>2. What happens to new images?</h2>
<p>For this test the VAE learns from 4,744 images, then reconstructs 256 held-out images. These pictures never enter a training batch. This is a separate, harder task than fitting the 50 familiar images above.</p>
<label for="seed">Training seed</label><select id="seed"><option value="0">Seed 0</option><option value="1">Seed 1</option><option value="2">Seed 2</option></select>
<label for="held-photo">Held-out photograph <output id="held-number">1</output> of 256</label><input id="held-photo" type="range" min="0" max="255" step="1" value="0">
<div class="pictures four"><figure><canvas id="held-original" width="32" height="32" role="img" aria-label="Original held-out image"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="held-old" width="32" height="32" role="img" aria-label="Earlier modified-decoder VAE reconstruction"></canvas><figcaption>Earlier decoder<br>Mean code</figcaption></figure><figure><canvas id="held-new" width="32" height="32" role="img" aria-label="Vanilla QVF VAE mean reconstruction"></canvas><figcaption>Vanilla QVF<br>Mean code</figcaption></figure><figure><canvas id="held-sample" width="32" height="32" role="img" aria-label="Vanilla QVF VAE sampled reconstruction"></canvas><figcaption>Vanilla QVF<br>Sampled code</figcaption></figure></div>
<p id="held-stats" class="muted stats"></p><p class="muted">All 256 validation images are available, with no per-image code fitting. Models are selected by the same validation objective. This validation set helps development; it is not an untouched final test set. Images are 32 × 32 pixels enlarged without smoothing.</p></div>
<div class="panel"><h2>3. Did restoring vanilla QVF improve held-out accuracy?</h2>
<div id="accuracy-chart" class="chart" role="img" aria-label="Paired-seed held-out PSNR for the earlier and vanilla decoders"></div><p id="accuracy" class="stats"></p><p id="interpretation"></p>
<p class="muted">Higher PSNR means lower squared pixel error. Points joined by a line have the same training seed. The encoder, KL penalty, 64-number code, data split, 7,680 optimizer updates and sampled coordinates are matched. Error bars in the text are sample standard deviations over three seeds, not confidence intervals.</p>
<div id="learning-chart" class="chart" role="img" aria-label="Held-out learning curve for the selected seed"></div><div class="legend"><span>Vanilla QVF VAE held-out PSNR</span></div><p class="muted">The learning curve follows the seed selected above. Its checkpoint-selection rule also includes sampled reconstruction error and KL, so the selected checkpoint need not maximize mean-code PSNR.</p></div>
<div class="panel"><h2>4. What is verified?</h2><p id="verification"></p><p>The decoder calls the audited QVF class directly. Output and gradient checks pass; reconstruction gradients reach the encoder's mean and variance heads and the circuit. Older checkpoints keep their original RGB interpretation.</p><p class="muted">This is a classical simulation of the quantum circuit. The experiment establishes the VAE connection and its measured reconstruction quality; it makes no quantum-advantage or hardware-scaling claim.</p></div>
<script>
const data=__DATA__,fit=data.fit50,large=data.larger,fmt=x=>x.toFixed(2),$=id=>document.getElementById(id),fitPhoto=$('fit-photo'),fitCode=$('fit-code'),heldPhoto=$('held-photo'),seed=$('seed');
const n=data.summary.vanilla_vae,o=data.summary.legacy_vae,g=data.summary.paired_gain;
$('fit-summary').textContent=`With an encoder-generated mean code, the final VAE reaches ${fmt(fit.train_final_mean_psnr_db)} dB on these familiar images (${fmt(fit.train_final_sampled_psnr_db)} dB with sampled codes). Vanilla QVF with directly optimized codes reached ${fmt(fit.vanilla_train_psnr_db)} dB.`;
$('fit-generalization').textContent=`This small-data model overfits: its validation-selected checkpoint reaches only ${fmt(fit.validation_mean_psnr_db)} dB on 256 other images. The close training fit demonstrates capacity, not reliable reconstruction of new photographs.`;
$('accuracy').textContent=`Earlier decoder: ${fmt(o.mean)} ± ${fmt(o.sample_sd)} dB. Vanilla QVF decoder: ${fmt(n.mean)} ± ${fmt(n.sample_sd)} dB. Mean paired change: ${g.mean>=0?'+':''}${g.mean.toFixed(3)} dB.`;
$('interpretation').textContent=Math.abs(g.mean)<.3?'Restoring vanilla QVF leaves held-out reconstruction accuracy close to the earlier result. The decoder is now faithful to the baseline, but that change alone has not solved the loss of fine detail on new images.':g.mean>0?'The restored vanilla decoder improves held-out accuracy under this matched protocol. The image panels show which details it preserves and where reconstruction errors remain.':'The restored vanilla decoder scores lower under this matched VAE protocol. Fidelity to the original QVF architecture does not by itself guarantee better encoder-based reconstruction.';
$('verification').textContent=`All ${data.verification.run_count} runs completed. Checkpoint reloads, raw reconstruction metrics, dataset IDs, source snapshots and checkpoint selection were checked. The original vanilla QVF source and baseline checkpoint remain unchanged.`;
function crop(id,img,i,group,columns,rows){const ctx=$(id).getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(img,1+(i%columns)*33,1+(Math.floor(i/columns)+group*rows)*33,32,32,0,0,32,32)}
function drawFit(){const i=Number(fitPhoto.value);crop('fit-original',fit.asset,i,0,10,5);crop('fit-direct',fit.asset,i,1,10,5);crop('fit-vae',fit.asset,i,fitCode.value==='mean'?2:3,10,5);$('fit-number').textContent=i+1;$('fit-stats').textContent=`Dataset image ${fit.ids[i]}. Directly optimized code: ${fmt(fit.per_image_vanilla_psnr_db[i])} dB; encoder mean: ${fmt(fit.per_image_vae_psnr_db[i])} dB.`;}
function drawHeld(){const i=Number(heldPhoto.value),r=large[Number(seed.value)];['held-original','held-old','held-new','held-sample'].forEach((id,j)=>crop(id,r.asset,i,j,16,16));$('held-number').textContent=i+1;$('held-stats').textContent=`Dataset image ${r.ids[i]}. Earlier decoder: ${fmt(r.per_image_legacy_psnr_db[i])} dB; vanilla decoder: ${fmt(r.per_image_vae_psnr_db[i])} dB. Both use encoder means.`;drawLearning();}
function axes(el,min,max){const w=el.clientWidth,h=225,L=43,R=18,T=22,B=40,sy=v=>h-B-(v-min)/(max-min)*(h-T-B);let parts='';for(let i=0;i<=4;i++){const v=min+(max-min)*i/4;parts+=`<line x1="${L}" x2="${w-R}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--border,#555)"/><text x="${L-7}" y="${sy(v)+4}" text-anchor="end" fill="var(--foreground,#ddd)" font-size="11">${v.toFixed(1)}</text>`;}return{w,h,L,R,T,B,sy,parts};}
function svg(el,a,extra){el.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${a.w} ${a.h}">${a.parts}${extra}</svg>`;}
function drawAccuracy(){const el=$('accuracy-chart'),values=large.flatMap(r=>[r.legacy_validation_psnr_db,r.validation_mean_psnr_db]),a=axes(el,Math.floor((Math.min(...values)-.15)*2)/2,Math.ceil((Math.max(...values)+.15)*2)/2),x1=a.L+(a.w-a.L-a.R)*.2,x2=a.L+(a.w-a.L-a.R)*.8;let parts='';large.forEach((r,i)=>{const c=`var(--chart-${i+1},#69a7ff)`,y1=a.sy(r.legacy_validation_psnr_db),y2=a.sy(r.validation_mean_psnr_db);parts+=`<line x1="${x1}" x2="${x2}" y1="${y1}" y2="${y2}" stroke="${c}" stroke-width="2" opacity=".65"/><circle cx="${x1}" cy="${y1}" r="4" fill="${c}"/><circle cx="${x2}" cy="${y2}" r="4" fill="${c}"/>`;});parts+=`<text x="${x1}" y="${a.h-15}" text-anchor="middle" fill="var(--foreground,#ddd)" font-size="12">Earlier decoder</text><text x="${x2}" y="${a.h-15}" text-anchor="middle" fill="var(--foreground,#ddd)" font-size="12">Vanilla QVF</text>`;svg(el,a,parts);}
function drawLearning(){const r=large[Number(seed.value)],el=$('learning-chart'),a=axes(el,0,Math.ceil(Math.max(...r.history.map(p=>p[1]))+1)),sx=v=>a.L+v/7680*(a.w-a.L-a.R),path=r.history.map((p,i)=>`${i?'L':'M'}${sx(p[0])},${a.sy(p[1])}`).join(' ');let parts=`<path d="${path}" stroke="var(--chart-1,#69a7ff)" stroke-width="2" fill="none"/>`;[0,3840,7680].forEach(v=>parts+=`<text x="${sx(v)}" y="${a.h-15}" text-anchor="middle" fill="var(--foreground,#ddd)" font-size="11">${v}</text>`);svg(el,a,parts);}
Promise.all([fit,...large].map((r,i)=>{r.asset=$(`asset-${i}`);return r.asset.decode()})).then(()=>{fitPhoto.addEventListener('input',drawFit);fitCode.addEventListener('change',drawFit);heldPhoto.addEventListener('input',drawHeld);seed.addEventListener('change',drawHeld);drawFit();drawHeld();drawAccuracy();let width=0;new ResizeObserver(()=>{const next=document.body.clientWidth;if(next!==width){width=next;drawAccuracy();drawLearning()}}).observe(document.body)}).catch(e=>console.error('Unable to load reconstructions',e));
</script></body></html>'''


def build(data, local_assets=False):
    assets = []
    for i, record in enumerate([data["fit50"], *data["larger"]]):
        path = Path(record["image"])
        source = str(path) if local_assets else "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        assets.append(f'<img id="asset-{i}" src="{html.escape(source, quote=True)}" alt="">')
    return HTML.replace("__ASSETS__", '<div hidden>' + ''.join(assets) + '</div>').replace(
        "__DATA__", json.dumps(data, allow_nan=False).replace("</", "<\\/"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    outputs = [(Path(str(args.output_stem) + ".html"), False),
               (Path(str(args.output_stem) + "_local.html"), True)]
    if any(path.exists() for path, _ in outputs):
        raise FileExistsError("Choose a fresh report output stem")
    data = json.loads(args.data.read_text())
    for path, local in outputs:
        with path.open("x") as handle:
            handle.write(build(data, local))
        print(path.resolve())


if __name__ == "__main__":
    main()
