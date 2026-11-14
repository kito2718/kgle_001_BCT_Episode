"""Fixed-shape T4 Division BigRU backend; no eager or build fallback."""
from pathlib import Path
import hashlib,json,os
import torch,tensorrt as trt

SHA={20260817:'13bf6766d13b02e1758a9164bfefe1fba0d365a8d0118aab352c677d558e4025',
     20260818:'158265a62d2c02a5441f5a35c62e457a66ffd19b738917d9ec202b66be5f0acb'}
CHECKPOINT={20260817:'28bf1bd594ffac5ed4c535f4d446eaa746e24002f8298bb38d9e7c5bf516d98f',
            20260818:'0607b9867fcd6f1c5e52fde5af3527587f0daaac6501349d86e56daf9b5c72dd'}

class TRTParent(torch.nn.Module):
    def __init__(self,seed,device):
        super().__init__()
        self.seed=seed;self.calls=0;self.padded_calls=0
        root=Path(os.environ['BIOHUB_DIVISION_TRT_ENGINE_DIR'])
        path=root/f'division_seed{seed}.engine'
        assert path.is_file(),path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA[seed]
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
        assert self.engine is not None
        self.context=self.engine.create_execution_context()
        assert tuple(self.context.get_tensor_shape('input'))==(32,3,14,12,48,48)
        shape=tuple(self.context.get_tensor_shape('binary'))
        dtype=torch.float16 if self.engine.get_tensor_dtype('binary')==trt.float16 else torch.float32
        self.output=torch.empty(shape,device=device,dtype=dtype)
        self.receipt=Path(os.environ['BIOHUB_DIVISION_TRT_RECEIPT_DIR'])/f'seed{seed}.json'
        self.receipt.parent.mkdir(parents=True,exist_ok=True)
        self.save()

    def save(self):
        self.receipt.write_text(json.dumps(dict(seed=self.seed,calls=self.calls,
            padded_calls=self.padded_calls,engine_sha256=SHA[self.seed],
            tensorrt=trt.__version__,shape=(32,3,14,12,48,48))))

    def forward(self,x):
        n=x.shape[0]
        assert 1<=n<=32 and tuple(x.shape[1:])==(3,14,12,48,48)
        if n<32:
            z=torch.zeros((32-n,*x.shape[1:]),device=x.device,dtype=x.dtype)
            x=torch.cat((x,z),dim=0)
            self.padded_calls+=1
        x=x.contiguous()
        with torch.inference_mode():
            self.context.set_tensor_address('input',x.data_ptr())
            self.context.set_tensor_address('binary',self.output.data_ptr())
            assert self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            result=self.output[:n].clone()
            self.calls+=1
            if self.calls==1 or self.calls%100==0:
                torch.cuda.synchronize()
                assert torch.isfinite(result).all()
                self.save()
            return {'binary':result}

def wrap_scorer(scorer):
    assert len(scorer.models)==2
    for index,seed in enumerate((20260817,20260818)):
        assert scorer.provenance[index]['seed']==seed
        assert scorer.provenance[index]['sha256']==CHECKPOINT[seed]
        scorer.models[index]=TRTParent(seed,scorer.device)
    return scorer
