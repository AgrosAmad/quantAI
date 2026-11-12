"""Render the verified vanilla QVF baseline, with every fitted image inspectable."""

import argparse
import base64
import html
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audited vanilla QVF reconstruction</title><style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground,#ddd);color-scheme:light dark}*{box-sizing:border-box}body{margin:0}p{line-height:1.55;margin:10px 0 14px}h2{font-size:20px;margin:0 0 12px}
.panel{padding:16px;background:var(--card,#222);border:1px solid var(--border,#444);border-radius:var(--radius,12px);margin:16px 0}.muted,figcaption{font-size:13px;line-height:1.5;color:var(--muted-foreground,#aaa)}a{color:var(--primary,#8ab4ff)}
label{display:block;margin:12px 0 6px;font-size:14px}select{width:100%;padding:8px;font:inherit;border:1px solid var(--border,#555);border-radius:6px;background:var(--background,#222);color:var(--foreground,#ddd)}input{width:100%;accent-color:var(--primary,#8ab4ff)}
.pictures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.two{grid-template-columns:repeat(2,minmax(0,1fr));max-width:480px;margin:0 auto}figure{margin:0}canvas{display:block;width:100%;height:auto;image-rendering:pixelated;border-radius:4px}figcaption{margin-top:7px}.chart{height:220px;width:100%}.chart svg{display:block;width:100%;height:100%}.stats{font-variant-numeric:tabular-nums}
@media(max-width:480px){h2{font-size:18px}.pictures{gap:7px}figcaption{font-size:12px}}
</style></head><body>__ASSETS__
<p><strong>Vanilla QVF can preserve fine image detail.</strong> These runs directly fit images with the original quantum model and optimized image codes. The model source is unchanged from the audited reproduction. There is no image encoder, VAE sampling, diffusion, or added RGB rescaling.</p>
<div class="panel"><h2>1. Fit one photograph closely</h2><p id="single-summary"></p><p class="muted">Some original CIFAR-10 photographs are already soft. The check here is whether reconstruction preserves the detail present in the source image.</p>
<label for="single">Photograph fitted by its own model</label><select id="single"></select>
<div class="pictures two"><figure><canvas id="single-original" width="32" height="32" role="img" aria-label="Original photograph"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="single-reconstruction" width="32" height="32" role="img" aria-label="Vanilla QVF single-image reconstruction"></canvas><figcaption>Vanilla QVF reconstruction</figcaption></figure></div>
<p id="single-stats" class="muted stats"></p><p class="muted">These are the first three CIFAR-10 dogs in dataset order, each fitted separately for 5,000 updates with seed 0. The pictures are 32 × 32 pixels, enlarged with no smoothing.</p></div>
<div class="panel"><h2>2. One shared model for 50 photographs</h2>
<p>Each photograph has its own optimized 64-number code. The amplitude network and five-qubit circuit are shared across all 50 photographs. Every update uses every pixel of the selected images.</p>
<label for="seed">Training seed</label><select id="seed"><option value="0">Seed 0</option><option value="1">Seed 1</option><option value="2">Seed 2</option></select>
<label for="photo">Photograph <output id="photo-number">1</output> of 50</label><input id="photo" type="range" min="0" max="49" step="1" value="0">
<div class="pictures"><figure><canvas id="original" width="32" height="32" role="img" aria-label="Original collection photograph"></canvas><figcaption>Original</figcaption></figure><figure><canvas id="released" width="32" height="32" role="img" aria-label="Released-code convention reconstruction"></canvas><figcaption>Released-code conventions</figcaption></figure><figure><canvas id="paper" width="32" height="32" role="img" aria-label="Paper-text convention reconstruction"></canvas><figcaption>Paper-text conventions</figcaption></figure></div>
<p id="collection-stats" class="muted stats"></p><p class="muted">The slider exposes all 50 fitted images, in dataset order. Each shared model uses 5,000 epochs, batches of 10 images, and 25,000 optimizer updates. These are reconstructions of training images.</p></div>
<div class="panel"><h2>3. Reconstruction accuracy</h2><div class="chart" id="chart" role="img" aria-label="Three-seed reconstruction PSNR for two vanilla QVF conventions with the paper's reported result as reference"></div>
<p id="numbers" class="stats"></p><p class="muted">Higher PSNR means lower pixel error. Dots are three-seed means; whiskers show one sample standard deviation. The shaded reference band is the paper's Gaussian + ReLU result, 29.8 ± 1.1 dB, from <a href="https://arxiv.org/html/2508.10900v2#S5.T3" target="_blank" rel="noopener">Table 3</a>. Our raw outputs are scored before clipping; clipping is used only to display valid RGB images.</p>
<p>The release and paper differ in their readout, coordinate encoding and code penalty. We kept those conventions separate. Both use the original circuit and no extra output rescaling.</p>
<p class="muted">The paper does not fully specify the image subset, code size, positional-encoding depth or batch size. This checks the architecture and a documented reconstruction protocol; matching a score does not recover every original experimental detail.</p></div>
<div class="panel"><h2>4. What the audit checked</h2><p id="audit"></p><p>The official source needs three rotation-index repairs and a dataset-subset reshape repair to run. The audit applies these in memory, preserving its circuit, amplitude network, RGB mapping and loss.</p>
<p id="reload"></p><p class="muted">This establishes the vanilla image-fitting baseline. It does not yet identify which change in the earlier encoder-based extension caused its loss of detail.</p></div>
<script>
const data=__DATA__,singleRuns=data.groups.vanilla_single,releaseRuns=data.groups.vanilla_collection,paperRuns=data.groups.vanilla_paper_text;
const all=[...singleRuns,...releaseRuns,...paperRuns],fmt=n=>n.toFixed(2),single=document.getElementById('single'),seed=document.getElementById('seed'),photo=document.getElementById('photo');
singleRuns.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`Photograph ${i+1} · dataset image ${r.dataset_indices[0]}`;single.appendChild(o)});
document.getElementById('single-summary').textContent=`The three individual fits reached ${Math.min(...singleRuns.map(r=>r.psnr_mean)).toFixed(1)}–${Math.max(...singleRuns.map(r=>r.psnr_mean)).toFixed(1)} dB. At this scale, the fitted pictures closely match their originals.`;
const a=data.summaries.vanilla_collection,b=data.summaries.vanilla_paper_text;
document.getElementById('numbers').textContent=`Released-code conventions: ${fmt(a.mean_psnr_db)} ± ${fmt(a.sample_sd_psnr_db)} dB. Paper-text conventions: ${fmt(b.mean_psnr_db)} ± ${fmt(b.sample_sd_psnr_db)} dB.`;
document.getElementById('audit').textContent='Twenty circuit configurations passed output, gradient and orthogonality checks against PennyLane. The full released model also matched initial predictions, gradients and 30 optimizer updates for both Gaussian and identity initialization.';
document.getElementById('reload').textContent=`All ${data.verification.run_count} final checkpoints were reloaded. Their raw reconstructions, dataset IDs and source hashes were checked, and trained-model outputs were compared with the released PennyLane implementation.`;
function crop(id,img,index,rowOffset,nrow){const canvas=document.getElementById(id),ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(img,1+(index%nrow)*33,1+(Math.floor(index/nrow)+rowOffset)*33,32,32,0,0,32,32)}
function drawSingle(){const r=singleRuns[Number(single.value)];crop('single-original',r.asset,0,0,1);crop('single-reconstruction',r.asset,0,1,1);document.getElementById('single-stats').textContent=`Raw reconstruction PSNR: ${fmt(r.psnr_mean)} dB. Higher PSNR means lower squared pixel error.`;}
function drawCollection(){const s=Number(seed.value),i=Number(photo.value),r=releaseRuns[s],p=paperRuns[s];crop('original',r.asset,i,0,10);crop('released',r.asset,i,5,10);crop('paper',p.asset,i,5,10);document.getElementById('photo-number').textContent=i+1;document.getElementById('collection-stats').textContent=`Dataset image ${r.dataset_indices[i]}. Released-code conventions: ${fmt(r.psnr_per_image[i])} dB. Paper-text conventions: ${fmt(p.psnr_per_image[i])} dB.`;}
function chart(){const el=document.getElementById('chart'),w=el.clientWidth,h=220,L=42,R=18,T=15,B=39,min=Math.floor(Math.min(28.7,a.mean_psnr_db-a.sample_sd_psnr_db,b.mean_psnr_db-b.sample_sd_psnr_db))-1,max=Math.ceil(Math.max(30.9,a.mean_psnr_db+a.sample_sd_psnr_db,b.mean_psnr_db+b.sample_sd_psnr_db))+1;
 const sy=y=>h-B-(y-min)/(max-min)*(h-T-B),xs=[L+(w-L-R)*.25,L+(w-L-R)*.75],fg='var(--foreground,#ddd)',grid='var(--border,#555)';let parts=`<rect x="${L}" y="${sy(30.9)}" width="${w-L-R}" height="${sy(28.7)-sy(30.9)}" fill="var(--chart-3,#93ca99)" opacity=".15"/>`;
 for(let i=0;i<=4;i++){const y=min+(max-min)*i/4;parts+=`<line x1="${L}" x2="${w-R}" y1="${sy(y)}" y2="${sy(y)}" stroke="${grid}" opacity=".5"/><text x="${L-7}" y="${sy(y)+4}" text-anchor="end" fill="${fg}" font-size="11">${y.toFixed(1)}</text>`;}
 [a,b].forEach((r,i)=>{const x=xs[i],y=sy(r.mean_psnr_db),lo=sy(r.mean_psnr_db-r.sample_sd_psnr_db),hi=sy(r.mean_psnr_db+r.sample_sd_psnr_db),c=`var(--chart-${i+1},#69a7ff)`;parts+=`<path d="M${x},${lo}V${hi}M${x-5},${lo}h10M${x-5},${hi}h10" stroke="${c}" stroke-width="2"/><circle cx="${x}" cy="${y}" r="5" fill="${c}"/><text x="${x}" y="${hi-10}" text-anchor="middle" fill="${fg}" font-size="12">${fmt(r.mean_psnr_db)}</text><text x="${x}" y="${h-14}" text-anchor="middle" fill="${fg}" font-size="11">${i===0?'Released code':'Paper text'}</text>`;});el.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">${parts}</svg>`;}
Promise.all(all.map((r,i)=>{r.asset=document.getElementById(`asset-${i}`);return r.asset.decode()})).then(()=>{single.addEventListener('change',drawSingle);seed.addEventListener('change',drawCollection);photo.addEventListener('input',drawCollection);drawSingle();drawCollection();chart();let width=0;new ResizeObserver(()=>{const next=document.body.clientWidth;if(next!==width){width=next;chart()}}).observe(document.body)}).catch(e=>console.error('Image loading failed',e));
</script></body></html>'''


def build(data, local_assets):
    rows = [row for group in ("vanilla_single", "vanilla_collection", "vanilla_paper_text") for row in data["groups"][group]]
    assets = []
    for i, row in enumerate(rows):
        path = Path(row["image_path"])
        source = str(path) if local_assets else "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        assets.append(f'<img id="asset-{i}" src="{html.escape(source, quote=True)}" alt="">')
    serialized = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    return TEMPLATE.replace("__ASSETS__", '<div hidden>' + ''.join(assets) + '</div>').replace("__DATA__", serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    outputs = [(Path(str(args.output_stem) + ".html"), False), (Path(str(args.output_stem) + "_local.html"), True)]
    if any(path.exists() for path, _ in outputs):
        raise FileExistsError("Choose a fresh output stem")
    for path, local_assets in outputs:
        with path.open("x") as handle:
            handle.write(build(data, local_assets))
        print(path.resolve())


if __name__ == "__main__":
    main()
