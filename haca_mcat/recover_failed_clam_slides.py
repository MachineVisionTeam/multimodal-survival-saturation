"""
Recover the 20 GBMLGG slides that CLAM's default segmentation rejected with
'Total number of contours to process: 0'.

Bypasses CLAM's CSV plumbing (which silently coerces booleans to strings)
and calls WholeSlideImage directly with progressively looser parameters.

Tier escalation per slide (stops at first tier that finds contours):
  T1: Otsu (proper bool)               + a_t= 16, a_h= 4
  T2: sthresh=4, use_otsu=False         + a_t= 16, a_h= 4
  T3: sthresh=2, use_otsu=False         + a_t=  4, a_h= 1
  T4: sthresh=1, use_otsu=False         + a_t=  1, a_h= 1
"""
import os, sys, json, time

CLAM_DIR = '/home/sbarua/Region_based_segmentation/genofilm_gbt_experiment/code/CLAM'
sys.path.insert(0, CLAM_DIR)
from wsi_core.WholeSlideImage import WholeSlideImage

FLAT_DIR     = '/home/sbarua/Region_based_segmentation/mcat_replication/gbmlgg_svs_flat'
PATCHES_DIR  = '/home/sbarua/Region_based_segmentation/mcat_replication/patches/TCGA_GBMLGG/patches'
LOG_PATH     = '/home/sbarua/Region_based_segmentation/mcat_replication/logs/clam_gbmlgg_recovery.log'

TIERS = [
    dict(name='T1_otsu',      use_otsu=True,  sthresh=8,
         filter_params=dict(a_t=16, a_h=4, max_n_holes=8)),
    dict(name='T2_sthresh4',  use_otsu=False, sthresh=4,
         filter_params=dict(a_t=16, a_h=4, max_n_holes=8)),
    dict(name='T3_sthresh2',  use_otsu=False, sthresh=2,
         filter_params=dict(a_t=4,  a_h=1, max_n_holes=8)),
    dict(name='T4_sthresh1',  use_otsu=False, sthresh=1,
         filter_params=dict(a_t=1,  a_h=1, max_n_holes=8)),
]

def find_missing():
    svs = {os.path.splitext(f)[0]: os.path.join(FLAT_DIR, f)
           for f in os.listdir(FLAT_DIR) if f.endswith('.svs')}
    h5  = {os.path.splitext(f)[0] for f in os.listdir(PATCHES_DIR) if f.endswith('.h5')}
    return {n: p for n, p in svs.items() if n not in h5}

def recover_slide(slide_path, log):
    name = os.path.splitext(os.path.basename(slide_path))[0]
    wsi = WholeSlideImage(slide_path)
    # default seg_level=-1 means pick the smallest available level
    seg_level = wsi.wsi.level_count - 1

    for tier in TIERS:
        try:
            wsi.segmentTissue(
                seg_level=seg_level,
                sthresh=tier['sthresh'],
                mthresh=7,
                close=4,
                use_otsu=tier['use_otsu'],
                filter_params=tier['filter_params'].copy(),
            )
        except Exception as e:
            log(f"  [{tier['name']}] segmentation crashed: {type(e).__name__}: {e}")
            continue

        n_contours = len(wsi.contours_tissue)
        log(f"  [{tier['name']}] seg_level={seg_level}  contours={n_contours}")
        if n_contours == 0:
            continue

        # Found contours — generate patches at level 0, 512x512
        try:
            wsi.process_contours(
                save_path=PATCHES_DIR,
                patch_level=0,
                patch_size=512,
                step_size=512,
                contour_fn='four_pt',
                use_padding=True,
            )
        except Exception as e:
            log(f"  [{tier['name']}] process_contours crashed: {type(e).__name__}: {e}")
            continue

        h5_path = os.path.join(PATCHES_DIR, name + '.h5')
        if os.path.isfile(h5_path):
            import h5py
            with h5py.File(h5_path, 'r') as h:
                n_patches = h['coords'].shape[0]
            log(f"  -> {tier['name']} SUCCESS: {n_patches} patches written")
            return tier['name'], n_patches
        else:
            log(f"  [{tier['name']}] process_contours ran but no .h5 file written")
    return None, 0

def main():
    missing = find_missing()
    print(f"Found {len(missing)} missing slides")
    out_lines = []
    summary = {}
    t_start = time.time()
    with open(LOG_PATH, 'w') as lf:
        def log(msg):
            lf.write(msg + '\n'); lf.flush()
            out_lines.append(msg)
            print(msg, flush=True)
        for i, (name, path) in enumerate(sorted(missing.items())):
            log(f"\n[{i+1}/{len(missing)}] {name}")
            tier, n = recover_slide(path, log)
            summary[name] = dict(tier=tier or 'ALL_FAILED', patches=n)
        log(f"\n=== RECOVERY SUMMARY ({time.time()-t_start:.1f}s) ===")
        succ = sum(1 for v in summary.values() if v['tier'] != 'ALL_FAILED')
        log(f"Recovered: {succ}/{len(summary)}")
        from collections import Counter
        c = Counter(v['tier'] for v in summary.values())
        log(f"Tier breakdown: {dict(c)}")
        for n, v in sorted(summary.items()):
            log(f"  {n[:70]:<70}  {v['tier']:>12}  {v['patches']:>8} patches")
    with open(LOG_PATH.replace('.log', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
