import json,unittest,numpy as np,torch
from pathlib import Path
from train_guided import Guided
class GuidedTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);cls.m=json.loads(Path('1-guided-model.json').read_text());cls.n=Guided();cls.n.load_state_dict({k:torch.tensor(v) for k,v in cls.m['weights'].items()});cls.n.eval()
 def test_fixtures(self):
  with torch.no_grad():
   for f in self.m['fixtures']:np.testing.assert_allclose(self.n(torch.tensor([f['input']]),torch.tensor([f['t']]),torch.tensor([f['label']]))[0].numpy(),f['output'],atol=1e-5)
 def test_fixed_budget(self):
  t=self.m['training'];self.assertEqual(t['additional_updates'],12000);self.assertEqual(len(t['history']),12);self.assertEqual(t['class_dropout'],.15);self.assertEqual(t['parameters'],sum(p.numel() for p in self.n.parameters()))
 def test_paired_audit(self):
  z=np.load('noise-probes.npz');a=json.loads(Path('noise-audit.json').read_text());old,new,ids=z['old'],z['new'],z['ids'];self.assertEqual(len(ids),2048);self.assertAlmostEqual(float((old-new).mean()),a['reduction']);unique=np.unique(ids);s=np.array([np.sum((old-new)[ids==i]) for i in unique]);counts=np.array([np.sum(ids==i) for i in unique]);rng=np.random.default_rng(321);v=[]
  for _ in range(3000):
   k=rng.integers(len(unique),size=len(unique));v.append(s[k].sum()/counts[k].sum())
  np.testing.assert_allclose(np.quantile(v,[.025,.975]),a['ci95']);self.assertGreater(a['ci95'][0],0)
if __name__=='__main__':unittest.main()
