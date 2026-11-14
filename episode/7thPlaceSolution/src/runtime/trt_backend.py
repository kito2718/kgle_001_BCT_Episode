"""Research nnUNet backend: static engines per shape, no eager fallback."""
import os,json,time,hashlib,copy
from pathlib import Path
import torch
class TensorRTNetwork(torch.nn.Module):
 def __init__(self,network,model_id):
  super().__init__();self.source=network;self.active=None;self.calls=0
  self.model_id=model_id;self.cache=Path(os.environ['BIOHUB_TRT_CACHE'])/model_id/os.environ.get('CUDA_VISIBLE_DEVICES','0');self.cache.mkdir(parents=True,exist_ok=True)
 def forward(self,x):
  import tensorrt as trt
  shape=tuple(x.shape);key='x'.join(map(str,shape));path=self.cache/(key+'.engine')
  with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
   if self.active!=key:
    self.context=None;self.engine=None;self.output=None
    if not path.exists():
     if os.environ.get('BIOHUB_TRT_REQUIRE_PREBUILT') == '1': raise RuntimeError(f'Missing prebuilt TensorRT engine: {path}')
     started=time.perf_counter()
     class Out(torch.nn.Module):
      def __init__(self,n):super().__init__();self.n=n
      def forward(self,x):
       y=self.n(x);return y[0] if isinstance(y,(list,tuple)) else y
     net=Out(copy.deepcopy(self.source).half().eval());example=x.contiguous().half()
     torch.onnx.export(net,example,str(path.with_suffix('.onnx')),input_names=['input'],output_names=['output'],opset_version=17,dynamo=False)
     del net,example;torch.cuda.empty_cache()
     logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(network,logger)
     assert parser.parse_from_file(str(path.with_suffix('.onnx'))),[str(parser.get_error(i)) for i in range(parser.num_errors)]
     cfg=builder.create_builder_config();cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30);cfg.set_flag(trt.BuilderFlag.FP16);cfg.builder_optimization_level=3
     serialized=builder.build_serialized_network(network,cfg);assert serialized is not None
     path.write_bytes(serialized)
     path.with_suffix('.json').write_text(json.dumps(dict(shape=shape,build_seconds=time.perf_counter()-started,tensorrt=trt.__version__,model_id=self.model_id,device=torch.cuda.get_device_name(x.device),engine_sha256=hashlib.sha256(bytes(serialized)).hexdigest())))
     del serialized,builder,network,parser,cfg
    meta=json.loads(path.with_suffix('.json').read_text());blob=path.read_bytes()
    assert meta['model_id']==self.model_id and meta['tensorrt']==trt.__version__
    assert meta['device']==torch.cuda.get_device_name(x.device) and meta['engine_sha256']==hashlib.sha256(blob).hexdigest()
    self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(blob);assert self.engine is not None
    self.context=self.engine.create_execution_context();self.output=torch.empty(tuple(self.context.get_tensor_shape('output')),device=x.device,dtype=torch.float16 if self.engine.get_tensor_dtype('output')==trt.float16 else torch.float32);self.active=key
   inp=x.contiguous().half();self.context.set_tensor_address('input',inp.data_ptr());self.context.set_tensor_address('output',self.output.data_ptr())
   assert self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
   result=self.output.clone();self.calls+=1
   if self.calls==1 or self.calls%100==0:
    torch.cuda.synchronize();assert torch.isfinite(result).all(),'Nonfinite TensorRT output'
    (self.cache/'calls.json').write_text(json.dumps(dict(calls=self.calls,last_shape=shape,pid=os.getpid())))
   return result
