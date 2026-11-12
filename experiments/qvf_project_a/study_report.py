"""Audit and summarize the reconstruction study, with an interactive HTML report."""

import argparse
import base64
import hashlib
import html
import json
import math
import statistics
from pathlib import Path


LABELS = {
    "baseline": "Original size + short training",
    "more_data": "More images + short training",
    "longer": "Original size + long training",
    "more_data_longer": "More images + long training",
    "latent128": "Larger code (128 numbers)",
    "wider": "Wider classical field network",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root, tag, refinement_prefix):
    reference = root / "20261108_v1_pilot_qvf_vae_seed0"
    reference_split = read(reference / "split.json")["indices"]
    reference_manifest = read(reference / "source_manifest.json")
    variants = []
    audited = []
    for name, label in LABELS.items():
        seeds = []
        for seed in range(3):
            directory = root / f"{tag}_reconstruction_{name}_seed{seed}"
            result, cfg = read(directory / "COMPLETE.json"), read(directory / "config.json")
            split = read(directory / "split.json")
            details = read(directory / "validation_details.json")
            indices = split["indices"]
            steps = 7680 if name in {"longer", "more_data_longer"} else 2560
            n_train = 4744 if name.startswith("more_data") else 1024
            assert result["status"] == "complete" and result["variant"] == name
            assert not (directory / "FAILED.json").exists()
            assert cfg["seed"] == seed and cfg["n_train"] == n_train and cfg["n_val"] == 256
            assert indices["validation"] == reference_split["validation"] == details["dataset_indices"]
            assert indices["train"][:1024] == reference_split["train"]
            assert len(set(indices["train"])) == n_train
            assert not set(indices["train"]) & set(indices["validation"])
            assert not split["official_test_set_used"]
            assert (cfg["model"]["qubits"], cfg["model"]["circuit_depth"]) == (5, 5)
            assert cfg["model"]["decoder"] == "qvf" and cfg["model"]["mode"] == "vae"
            assert cfg["batch_size"] == 32 and cfg["pixels_per_image"] == 256
            assert cfg["vae_max_steps"] == steps and cfg["vae_drop_last"]
            assert cfg["kl_warmup_steps"] == cfg["vae_eval_every_steps"] == 320
            assert cfg["kl_weight"] == 1e-5 and cfg["device"] == "cuda:0"
            vae = result["vae"]
            assert vae["optimizer_steps"] == steps
            assert vae["training_image_observations"] == steps * 32
            assert vae["training_coordinate_queries"] == steps * 32 * 256
            assert vae["unique_training_images_seen"] == n_train
            assert math.isclose(statistics.mean(details["mean_code_psnr_db"]),
                                vae["validation"]["mean_code"]["psnr_db"], abs_tol=5e-6)
            manifest = read(directory / "source_manifest.json")
            quantum_file = "experiments/qvf_repro/qvf_fast.py"
            assert manifest[quantum_file] == reference_manifest[quantum_file]
            for filename, expected in manifest.items():
                assert digest(directory / "source_snapshot" / filename) == expected
            history = [json.loads(line) for line in (directory / "vae/history.jsonl").read_text().splitlines()]
            validation = [row for row in history if "validation" in row]
            assert [row["step"] for row in validation] == [1] + list(range(320, steps + 1, 320))
            selected = min(validation, key=lambda row: row["validation"]["sampled_code"]["mse"]
                           + cfg["kl_weight"] * row["validation"]["kl_nats_per_image"])
            assert selected["step"] == vae["best_step"]
            item = {"seed": seed, "psnr_db": vae["validation"]["mean_code"]["psnr_db"],
                    "mse": vae["validation"]["mean_code"]["mse"],
                    "sampled_psnr_db": vae["validation"]["sampled_code"]["psnr_db"],
                    "gradient_mse": result["validation_gradient_mse"], "best_step": vae["best_step"],
                    "training_subset_psnr_db": vae["training_subset"]["mean_code"]["psnr_db"],
                    "history": [[0, vae["initial_validation"]["mean_code"]["psnr_db"]]]
                               + [[row["step"], row["validation"]["mean_code"]["psnr_db"]] for row in validation]}
            seeds.append(item)
            audited.append(directory.name)
            if seed == 0:
                visual = {"image": str((directory / "vae/validation_reconstruction.png").resolve()),
                          "image_psnr_db": details["mean_code_psnr_db"][:16]}
        values = [row["psnr_db"] for row in seeds]
        variants.append({"name": name, "label": label, "n_train": n_train, "steps": steps,
                         "latent_dim": cfg["model"]["latent_dim"], "field_width": cfg["model"]["field_width"],
                         "parameters": vae["parameter_counts"], "mean_psnr_db": statistics.mean(values),
                         "sd_psnr_db": statistics.stdev(values), "mean_mse": statistics.mean(row["mse"] for row in seeds),
                         "seeds": seeds, **visual})

    lookup = {row["name"]: row for row in variants}
    comparisons = []
    for a, b in [("more_data", "baseline"), ("more_data_longer", "longer"),
                 ("longer", "baseline"), ("more_data_longer", "baseline"),
                 ("latent128", "baseline"), ("wider", "baseline")]:
        gains = [x["psnr_db"] - y["psnr_db"] for x, y in zip(lookup[a]["seeds"], lookup[b]["seeds"])]
        comparisons.append({"treatment": a, "reference": b, "paired_seed_gains_db": gains,
                            "mean_gain_db": statistics.mean(gains), "sd_gain_db": statistics.stdev(gains)})
    # The longer runs must extend exactly the same trajectory as the short runs.
    for short, long in [("baseline", "longer"), ("more_data", "more_data_longer")]:
        for a, b in zip(lookup[short]["seeds"], lookup[long]["seeds"]):
            assert a["history"] == b["history"][:len(a["history"])]

    refinements = []
    for suffix, label in [("original", "Original pilot · pixel error only"),
                          ("original_prior", "Original pilot · with mean-code prior penalty"),
                          ("larger", "More images + long training · pixel error only")]:
        directory = root / f"{refinement_prefix}_{suffix}"
        result, cfg = read(directory / "COMPLETE.json"), read(directory / "config.json")
        details = read(directory / "per_image.json")
        assert result["status"] == "complete" and result["model_weights_unchanged"] and result["checkpoint_unchanged"]
        assert details["dataset_indices"] == reference_split["validation"]
        assert cfg["steps"] == 400 and cfg["learning_rate"] == 0.03
        assert digest(Path(cfg["source_run"]) / "vae/checkpoints/best.pt") == cfg["checkpoint_sha256"]
        refinements.append({"name": suffix, "label": label, "result": result, "prior_weight": cfg["prior_weight"],
                            "image": str((directory / "reconstruction.png").resolve()),
                            "history": read(directory / "history.json"),
                            "initial_psnr_db": details["initial_psnr_db"][:16],
                            "refined_psnr_db": details["refined_psnr_db"][:16]})
    verification = {"status": "passed", "runs_audited": audited, "run_count": len(audited),
                    "identical_validation_ids": True, "validation_count": 256,
                    "original_training_ids_preserved": True, "no_train_validation_overlap": True,
                    "all_training_images_seen": True, "optimizer_and_coordinate_budgets_match": True,
                    "long_runs_extend_short_trajectories_exactly": True,
                    "checkpoint_selection_matches_declared_objective": True,
                    "source_snapshot_hashes_match": True, "original_quantum_simulator_unchanged": True,
                    "refinement_source_checkpoints_unchanged": True,
                    "official_test_set_used": False, "diffusion_retrained": False}
    return {"variants": variants, "comparisons": comparisons, "refinements": refinements,
            "validation_ids": reference_split["validation"][:16], "verification": verification,
            "uncertainty": "Sample standard deviation across three training seeds on one fixed validation split; not a confidence interval."}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project A: data size, reconstruction and code fitting</title>
<style>
:root{font-family:var(--font-sans,system-ui);color:var(--foreground,#ddd);color-scheme:light dark}
*{box-sizing:border-box}body{margin:0}p{line-height:1.55;margin:10px 0 14px}h2{font-size:20px;margin:0 0 12px}h3{font-size:16px;margin:0 0 10px}
.panel{padding:16px;background:var(--card,#202127);border:1px solid var(--border,#444);border-radius:var(--radius,12px);margin:16px 0}
.muted,figcaption{color:var(--muted-foreground,#aaa);font-size:13px;line-height:1.5}.plot{height:220px;width:100%}.plot svg{display:block;width:100%;height:100%}
label{display:block;font-size:14px;margin:12px 0 6px}select{font:inherit;color:var(--foreground,#ddd);background:var(--background,#202127);border:1px solid var(--border,#555);border-radius:6px;padding:8px;width:100%}
input[type=range]{width:100%;accent-color:var(--primary,#69a7ff);cursor:pointer}.triptych{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:12px}figure{margin:0}
canvas{display:block;width:100%;height:auto;image-rendering:pixelated;border-radius:4px}figcaption{margin-top:6px}.metrics{font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;margin-top:10px}.legend span:before{content:"";display:inline-block;width:12px;height:3px;background:var(--chart-1,#69a7ff);margin-right:4px;vertical-align:middle}.legend span:nth-child(2):before{background:var(--chart-2,#f3b562)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px 5px;text-align:right;border-bottom:1px solid var(--border,#444)}td:first-child,th:first-child{text-align:left}th{font-weight:600}.table-wrap{overflow-x:auto}details{margin-top:14px}summary{cursor:pointer;font-size:14px}
@media(max-width:480px){h2{font-size:18px}.triptych{gap:7px}figcaption{font-size:12px}}
</style></head><body>__ASSETS__
<p><strong>Yes—dataset size is part of the problem.</strong> Across three training seeds, more images helped substantially once the model had enough training. The reconstructions still lose fine detail.</p>
<div class="panel"><h2>1. More images need enough training</h2>
<p>We increased training from <strong>1,024 to 4,744 dog photographs</strong>, keeping the same 256 validation photographs separate. A training step still sees 32 images and 256 pixel locations per image. The circuit stays at five qubits and depth five.</p>
<div class="legend"><span>1,024 training images</span><span>4,744 training images</span></div>
<div class="plot" id="data-chart" role="img" aria-label="Validation reconstruction PSNR by training budget for small and larger datasets"></div>
<p class="muted">Vertical axis: reconstruction PSNR in dB; higher is better. Dots show the three-seed mean, whiskers show ± one sample standard deviation. These are selected checkpoints within each training budget. The connecting lines join two tested budgets.</p>
<p id="data-conclusion"></p>
<details><summary>All six settings: exact results and controls</summary><div class="table-wrap"><table><thead><tr><th>Setting</th><th>Images</th><th>Steps</th><th>PSNR, mean ± SD</th></tr></thead><tbody id="results-table"></tbody></table></div>
<p class="muted">The wider network and larger code change classical parameter counts and compute. They keep five qubits and depth five. Every setting uses seeds 0, 1 and 2. These measurements use one validation split for checkpoint selection; they are not a final test-set assessment.</p></details>
</div>
<div class="panel"><h2>2. Look at the same photograph</h2>
<p>The encoder sees a photograph once, produces its code, and QVF reconstructs the pixels. These are reconstructions of known input photographs, not new diffusion samples.</p>
<label for="variant">Compare the original-size baseline with</label><select id="variant"></select>
<label for="photo">Validation photograph <output id="photo-number"></output> of 16</label><input id="photo" type="range" min="0" max="15" value="0" step="1">
<div class="triptych"><figure><canvas id="photo-original" width="32" height="32" role="img" aria-label="Original photograph"></canvas><figcaption>Original photograph</figcaption></figure><figure><canvas id="photo-baseline" width="32" height="32" role="img" aria-label="Original-size baseline reconstruction"></canvas><figcaption>Original size, short training</figcaption></figure><figure><canvas id="photo-selected" width="32" height="32" role="img" aria-label="Selected setting reconstruction"></canvas><figcaption id="selected-caption"></figcaption></figure></div>
<p class="muted metrics" id="photo-stats"></p>
<div class="legend"><span>Original-size baseline</span><span>Selected setting</span></div>
<div class="plot" id="training-chart" role="img" aria-label="Validation PSNR during training for seed zero"></div>
<p class="muted">Horizontal axis: training steps. The dotted marker identifies the selected checkpoint. All pictures and this learning curve use seed 0, and the slider follows the first 16 validation images in their saved order. The images are only 32 × 32 pixels, enlarged here for inspection.</p>
</div>
<div class="panel"><h2>3. Can QVF do more with a better code?</h2>
<p>Here we freeze the encoder and QVF weights. Starting from the encoder's code, we adjust <strong>only that image's code</strong> for 400 optimization steps, using the target photograph to measure its pixel error.</p>
<p id="refinement-conclusion"></p>
<label for="refinement">Code-fitting experiment</label><select id="refinement"></select>
<label for="fit-photo">Validation photograph <output id="fit-number"></output> of 16</label><input id="fit-photo" type="range" min="0" max="15" value="0" step="1">
<div class="triptych"><figure><canvas id="fit-original" width="32" height="32" role="img" aria-label="Target photograph for code fitting"></canvas><figcaption>Target photograph</figcaption></figure><figure><canvas id="fit-encoded" width="32" height="32" role="img" aria-label="Reconstruction from encoder code"></canvas><figcaption>Encoder's code</figcaption></figure><figure><canvas id="fit-refined" width="32" height="32" role="img" aria-label="Reconstruction from individually fitted code"></canvas><figcaption>Individually fitted code</figcaption></figure></div>
<p class="muted metrics" id="fit-stats"></p>
<div class="plot" id="refinement-chart" role="img" aria-label="Mean validation reconstruction PSNR during image-specific code fitting"></div>
<p class="muted" id="fit-description"></p>
<p>This reveals room to improve the encoder and its training. It does not prove the fitted codes are optimal. The fitted results use the target image repeatedly, so they are a diagnostic—not the quality of one-pass encoding or unconditional generation.</p>
</div>
<p><strong>What this changes:</strong> the strongest tested representation uses more data and longer training with the same circuit size. It is a better candidate for the next diffusion experiment. Diffusion has not been retrained in this study, so improved generation remains to be tested.</p>
<p class="muted">All 18 runs passed the split, training-budget, checkpoint-selection and source-snapshot checks. Code fitting left the model weights and source checkpoints unchanged. Exact classical simulation throughout; no hardware or quantum-advantage claim.</p>
<script>
const data=__DATA__,variants=data.variants,refinements=data.refinements;
const colors=['var(--chart-1,#69a7ff)','var(--chart-2,#f3b562)'],textColor='var(--foreground,#ddd)',gridColor='var(--border,#555)';
const byName=Object.fromEntries(variants.map(r=>[r.name,r])),baseline=byName.baseline;
const variant=document.getElementById('variant'),refinement=document.getElementById('refinement');
variants.forEach((v,i)=>{const option=document.createElement('option');option.value=i;option.textContent=v.label;variant.appendChild(option)});
variant.value=variants.findIndex(v=>v.name==='more_data_longer');
refinements.forEach((v,i)=>{const option=document.createElement('option');option.value=i;option.textContent=v.label;refinement.appendChild(option)});
const fmt=(n,d=2)=>n.toFixed(d);
document.getElementById('data-conclusion').textContent=`At 2,560 steps, adding images changed mean PSNR from ${fmt(baseline.mean_psnr_db)} to ${fmt(byName.more_data.mean_psnr_db)} dB—little improvement. At 7,680 steps, it improved from ${fmt(byName.longer.mean_psnr_db)} to ${fmt(byName.more_data_longer.mean_psnr_db)} dB. The larger dataset helped in all three seeds at the longer budget.`;
document.getElementById('refinement-conclusion').textContent=`On the original pilot, fitting codes raised average reconstruction from ${fmt(refinements[0].result.initial_psnr_db)} to ${fmt(refinements[0].result.refined_psnr_db)} dB. After more data and longer training, the gap was smaller: ${fmt(refinements[2].result.initial_psnr_db)} to ${fmt(refinements[2].result.refined_psnr_db)} dB. These diagnostics use seed 0.`;
document.getElementById('results-table').innerHTML=variants.map(v=>`<tr><td>${v.label}</td><td>${v.n_train.toLocaleString()}</td><td>${v.steps.toLocaleString()}</td><td>${fmt(v.mean_psnr_db)} ± ${fmt(v.sd_psnr_db)}</td></tr>`).join('');
function txt(x,y,s,anchor='start',size=11){return `<text x="${x}" y="${y}" fill="${textColor}" text-anchor="${anchor}" font-size="${size}" font-family="system-ui">${s}</text>`}
function plot(id,series,opts){const el=document.getElementById(id),w=el.clientWidth,h=220,L=42,R=18,T=16,B=40,xmin=opts.xmin??0,xmax=opts.xmax,ymin=opts.ymin,ymax=opts.ymax;
 const sx=x=>L+(x-xmin)/(xmax-xmin)*(w-L-R),sy=y=>h-B-(y-ymin)/(ymax-ymin)*(h-T-B);let body='';
 for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;body+=`<line x1="${L}" x2="${w-R}" y1="${sy(y)}" y2="${sy(y)}" stroke="${gridColor}" opacity=".5"/>`+txt(L-6,sy(y)+4,fmt(y,1),'end')}
 for(const x of opts.xticks??[0,xmax/2,xmax])body+=txt(sx(x),h-18,Math.round(x).toLocaleString(),'middle');
 series.forEach((points,i)=>{body+=`<polyline fill="none" stroke="${colors[i]}" stroke-width="2.5" points="${points.map(p=>`${sx(p[0])},${sy(p[1])}`).join(' ')}"/>`;
 points.forEach(p=>{if(p.length>2){body+=`<path d="M${sx(p[0])},${sy(p[1]-p[2])}V${sy(p[1]+p[2])}M${sx(p[0])-5},${sy(p[1]-p[2])}h10M${sx(p[0])-5},${sy(p[1]+p[2])}h10" stroke="${colors[i]}" stroke-width="2"/><circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="${colors[i]}"><title>${fmt(p[1],3)} ± ${fmt(p[2],3)} dB</title></circle>`}})});
 if(opts.marker!==undefined)body+=`<line x1="${sx(opts.marker)}" x2="${sx(opts.marker)}" y1="${T}" y2="${h-B}" stroke="${textColor}" opacity=".6" stroke-dasharray="3 4"/>`;
 el.innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">${body}</svg>`;
}
function crop(id,image,idx,row){const canvas=document.getElementById(id),ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(image,2+idx*34,2+row*34,32,32,0,0,32,32)}
function updatePhotos(){const v=variants[Number(variant.value)],idx=Number(document.getElementById('photo').value);
 document.getElementById('photo-number').textContent=idx+1;document.getElementById('selected-caption').textContent=v.label;
 crop('photo-original',baseline.asset,idx,0);crop('photo-baseline',baseline.asset,idx,1);crop('photo-selected',v.asset,idx,1);
 document.getElementById('photo-stats').textContent=`Dataset image ${data.validation_ids[idx]}. Baseline: ${fmt(baseline.image_psnr_db[idx])} dB. Selected: ${fmt(v.image_psnr_db[idx])} dB. Both use the encoder mean code.`;
 plot('training-chart',[baseline.seeds[0].history,v.seeds[0].history],{xmax:v.steps,ymin:12,ymax:22,marker:v.seeds[0].best_step});
}
function updateFitting(){const v=refinements[Number(refinement.value)],idx=Number(document.getElementById('fit-photo').value);
 document.getElementById('fit-number').textContent=idx+1;crop('fit-original',v.asset,idx,0);crop('fit-encoded',v.asset,idx,1);crop('fit-refined',v.asset,idx,2);
 document.getElementById('fit-stats').textContent=`This image: ${fmt(v.initial_psnr_db[idx])} → ${fmt(v.refined_psnr_db[idx])} dB. All 256 images: ${fmt(v.result.initial_psnr_db)} → ${fmt(v.result.refined_psnr_db)} dB.`;
 plot('refinement-chart',[v.history.map(p=>[p.step,p.best_psnr])],{xmax:400,ymin:19,ymax:22,xticks:[0,100,200,300,400]});
 document.getElementById('fit-description').textContent='Horizontal axis: image-specific fitting steps. Vertical axis: average PSNR of the best codes so far. '+(v.prior_weight ? 'This includes a 1e−5 penalty on half the squared code norm, corresponding to the mean-dependent part of the Gaussian KL. It is not the full stochastic VAE objective.' : 'The fitting objective is deterministic pixel error; it differs from the stochastic VAE training objective.');
}
function draw(){plot('data-chart',[[baseline,byName.longer],[byName.more_data,byName.more_data_longer]].map(rows=>rows.map(r=>[r.steps,r.mean_psnr_db,r.sd_psnr_db])),{xmin:1500,xmax:8750,xticks:[2560,7680],ymin:19,ymax:21});updatePhotos();updateFitting()}
const assets=[...variants,...refinements];
Promise.all(assets.map((v,i)=>{const img=document.getElementById(`asset-${i}`);v.asset=img;return img.decode()})).then(()=>{
 variant.addEventListener('change',updatePhotos);refinement.addEventListener('change',updateFitting);
 document.getElementById('photo').addEventListener('input',updatePhotos);document.getElementById('fit-photo').addEventListener('input',updateFitting);
 let oldWidth=0;new ResizeObserver(()=>{const width=document.body.clientWidth;if(width!==oldWidth){oldWidth=width;draw()}}).observe(document.body);draw();
}).catch(e=>{console.error('Image asset failed',e);document.getElementById('photo-stats').textContent='Could not load a local image. Open the portable HTML report.'});
</script></body></html>'''


def render(data, embed_images):
    assets = []
    for index, row in enumerate(data["variants"] + data["refinements"]):
        path = Path(row["image"])
        source = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if embed_images else str(path)
        assets.append(f'<img id="asset-{index}" src="{html.escape(source, quote=True)}" alt="">')
    serialized = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    return HTML.replace("__ASSETS__", '<div hidden>' + ''.join(assets) + '</div>').replace("__DATA__", serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--tag", default="20261108_v2")
    parser.add_argument("--refinement-prefix", default="20261108_reconstruction_v2_refine")
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    data = summarize(args.results.resolve(), args.tag, args.refinement_prefix)
    outputs = {".json": json.dumps(data, indent=2, allow_nan=False) + "\n",
               "_verification.json": json.dumps(data["verification"], indent=2) + "\n",
               ".html": render(data, True), "_local.html": render(data, False)}
    paths = [Path(str(args.output_stem) + suffix) for suffix in outputs]
    if any(path.exists() for path in paths):
        raise FileExistsError("Choose a fresh output stem; existing study reports are preserved.")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in zip(paths, outputs.values()):
        with path.open("x") as handle:
            handle.write(contents)
        print(path.resolve())


if __name__ == "__main__":
    main()
