"""Original1500 scientific configuration, 2000 epoch budget, per-sample logs."""
import os,json,csv,time
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.variants.biohub.nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculum import nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculumXYHalfIter250Long1500
class nnUNetTrainerBiohubColabM2000(nnUNetTrainerBiohubTemporalDilatedBackwardMotionCurriculumXYHalfIter250Long1500):
    # Keep 100-epoch milestones after 1500 for Drive-backed SWA candidates.
    milestone_epochs=frozenset(range(1600,4001,100)) | frozenset((1000,1500))
    def initialize(self):
        super().initialize()
        self._sample_context=None
        self.loss.register_forward_hook(self._sample_hook)
    def _sample_hook(self,module,args,result):
        if self._sample_context is None:return
        split,keys=self._sample_context
        out,target=args
        rows=[]
        heat,motion,vox=module.last_heatmap_loss,module.last_motion_loss,module.last_motion_voxels
        with torch.no_grad():
            for i,key in enumerate(keys):
                loss=module.forward(out[i:i+1].detach(),target[i:i+1])
                rows.append(dict(epoch=int(self.current_epoch),split=split,sample_id=str(key),series_id=str(key).rsplit('_t',1)[0],fold=str(self.fold),label=int((target[i,0]==1).sum().item()),prediction=float(torch.sigmoid(out[i,1].float()-out[i,0].float()).mean()),loss=float(loss),heatmap_loss=module.last_heatmap_loss,motion_loss=module.last_motion_loss))
        module.last_heatmap_loss,module.last_motion_loss,module.last_motion_voxels=heat,motion,vox
        p=Path(os.environ['BIOHUB_OUTPUTS_DIR'])/'sample_losses'/f'{split}_sample_loss_by_epoch.csv';p.parent.mkdir(exist_ok=True,parents=True)
        exists=p.exists()
        with p.open('a') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));
            if not exists:w.writeheader()
            w.writerows(rows)
    def train_step(self,batch):
        self._sample_context=('train',batch['keys'])
        try:return super().train_step(batch)
        finally:self._sample_context=None
    def validation_step(self,batch):
        self._sample_context=('valid',batch['keys'])
        try:return super().validation_step(batch)
        finally:self._sample_context=None
    def on_epoch_end(self):
        super().on_epoch_end()
        p=Path(os.environ['BIOHUB_OUTPUTS_DIR'])/'progress.json'
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(completed_epochs=self.current_epoch,total_epochs=self.num_epochs,updated_at=time.time(),output_folder=self.output_folder)));tmp.replace(p)
