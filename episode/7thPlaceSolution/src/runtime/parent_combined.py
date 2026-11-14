"""Batched fp32 input construction on GPU for unaugmented parent inference."""
import math,numpy as np,torch

def predict(self,raw,heat):
 device=self.device
 r=torch.as_tensor(np.stack(raw)[:,2:7],dtype=torch.float32,device=device)
 h=torch.as_tensor(np.stack(heat),dtype=torch.float32,device=device)
 def channels(t):
  back=(t[:,1:2]-t[:,0:1])/math.sqrt(2.);forward=(t[:,2:3]-t[:,1:2])/math.sqrt(2.)
  return torch.cat((t,back,forward,back.abs(),forward.abs()),dim=1)
 raw_channels=torch.stack([channels(r[:,i:i+3]) for i in range(3)],dim=1)
 mean=self.mean.to(device)[None,None,:,None,None,None];std=self.std.to(device)[None,None,:,None,None,None]
 raw_channels=(raw_channels-mean)/std;hh=channels(h)[:,None].expand(-1,3,-1,-1,-1,-1)
 images=torch.cat((raw_channels,hh),dim=2)
 with torch.inference_mode():result=torch.stack([torch.sigmoid(model(images)['binary'].float()).reshape(-1) for model in self.models],dim=1).cpu().numpy()
 if not np.isfinite(result).all():raise RuntimeError('Nonfinite vector parent')
 return result
