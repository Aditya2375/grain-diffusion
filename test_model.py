import unittest,json,hashlib
from pathlib import Path
import numpy as np,torch
from train import Net,schedule
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);cls.m=json.loads(Path('1-model.json').read_text());cls.net=Net();cls.net.load_state_dict({k:torch.tensor(v) for k,v in cls.m['weights'].items()});cls.net.eval()
 def test_fixtures(self):
  with torch.no_grad():
   for f in self.m['fixtures']:
    actual=self.net(torch.tensor([f['input']]),torch.tensor([f['t']]),torch.tensor([f['label']]))[0].numpy();np.testing.assert_allclose(actual,f['output'],atol=1e-5)
 def test_schedule(self):
  b,a=schedule();np.testing.assert_allclose(b,self.m['beta']);np.testing.assert_allclose(a,self.m['alpha_bar']);self.assertTrue(np.all(np.diff(a)<0));self.assertTrue(np.all((b>0)&(b<1)))
 def test_source(self):
  for name,h in self.m['training']['sha256'].items():self.assertEqual(hashlib.sha256((Path('data')/name).read_bytes()).hexdigest(),h)
 def test_training(self):
  h=self.m['training']['history'];self.assertEqual(h[-1]['step'],30000);self.assertLess(h[-1]['validation'],h[0]['validation']);self.assertEqual(self.m['training']['train_images']+self.m['training']['validation_images'],60000)
if __name__=='__main__':unittest.main()
