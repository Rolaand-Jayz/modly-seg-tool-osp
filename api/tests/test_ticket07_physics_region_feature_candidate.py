from __future__ import annotations
import sys, unittest
import importlib.util
from pathlib import Path
import numpy as np

ADAPTER=Path(__file__).resolve().parents[1]/"runtime"/"adapters"/"material-identity"
sys.path.insert(0,str(ADAPTER))
if importlib.util.find_spec("PIL") is not None:
    from PIL import Image
    from dinov2_evaluator import development_truth_plan
    from evaluator import SUPPORTED_LABELS
    from physics_region_feature_candidate import _region_features
    from physics_feature_candidate import _scores
else:
    Image = None
    development_truth_plan = None
    SUPPORTED_LABELS = ()
    _region_features = _scores = None

@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "rejected research candidate requires optional Pillow")
class Ticket07MaskedPhysicsFeatures(unittest.TestCase):
    def test_pixels_outside_topology_mask_do_not_affect_features(self):
        mask=np.zeros((20,20),dtype=bool); mask[5:15,5:15]=True
        a=np.full((20,20,3),127,dtype=np.uint8); a[mask]=[180,40,20]
        b=a.copy(); b[~mask]=[0,255,70]
        self.assertEqual(_region_features(Image.fromarray(a),mask),_region_features(Image.fromarray(b),mask))

    def test_oof_is_five_fold_object_disjoint_dev_only(self):
        plan=development_truth_plan(); rows=[]
        for cid,t in plan.items():
            for vi in range(4):
                f=[0.]*30
                if t['truth_label'] in SUPPORTED_LABELS: f[SUPPORTED_LABELS.index(t['truth_label'])]=1.
                rows.append({'case_id':cid,'object_id':t['object_id'],'region_id':'r','view_id':str(vi),'crop_input_digest':cid+str(vi),'features':f})
        scored=_scores(rows,plan)
        self.assertEqual(len(scored),140)
        for r in scored: self.assertEqual(r['fold'],plan[r['case_id']]['fold'])

if __name__=='__main__': unittest.main()
