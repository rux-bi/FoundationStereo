import warnings, argparse, logging, os, sys, imageio
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import omegaconf, yaml, torch, pdb
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo
import torch.nn as nn
from core.utils.utils import InputPadder
import onnxruntime as ort
import numpy as np
from torch.utils.benchmark import Timer, Measurement
from tools.symbolic_shape_infer import SymbolicShapeInference
import onnx
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
from core.geometry import Combined_Geo_Encoding_Volume
from core.update import BasicSelectiveMultiUpdateBlock
from core.submodule import context_upsample, Conv2x
import torch.nn.functional as F
# Add numpy.core.multiarray.scalar to the safe globals list right at the start
add_safe_globals([scalar])
autocast = torch.cuda.amp.autocast


class FoundationStereoOnnx(nn.Module):
    def __init__(self, args, valid_iters):
        super().__init__()
        assets = torch.load("/offboard/FoundationStereo/assets/simplified_onnx_inputs.pth")
        self.args = args
        self.corr_levels = 2
        self.dx = assets["dx"]
        self.valid_iters = valid_iters
        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )
        volume_dim = 28
        self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        # self.padder = InputPadder(img_shape, divis_by=32, force_square=False)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp.float()

    @torch.no_grad()
    def forward(self, init_disp):
        """ Removes extra outputs and hyper-parameters """
        disp = init_disp.float().clone()
        left = torch.randn(1, 128, 96, 160).cuda().float()
        right = torch.randn(1, 128, 96, 160).cuda().float()
        volume = torch.randn(1, 28, 104, 96, 160).cuda().float()
        b, c, h, w = left.shape
        coords = torch.arange(w, dtype=torch.float, device=disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        geo_fn = Combined_Geo_Encoding_Volume(left, right, volume, num_levels=self.corr_levels, dx=self.dx)
        inp_list = [
            torch.randn(1, 128, 96, 160).cuda().float(),
            torch.randn(1, 128, 48, 80).cuda().float(),
            torch.randn(1, 128, 24, 40).cuda().float(),
        ]
        net_list = [
            torch.randn(1, 128, 96, 160).cuda().float(),
            torch.randn(1, 128, 48, 80).cuda().float(),
            torch.randn(1, 128, 24, 40).cuda().float(),
        ]
        att = [
            torch.randn(1, 1, 96, 160).cuda().float(),
            torch.randn(1, 1, 48, 80).cuda().float(),
            torch.randn(1, 1, 24, 40).cuda().float(),
        ]
        stem_2x = torch.randn(1, 32, 192, 320).cuda().float()
        disp_preds = []   
        
        for itr in range(self.valid_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, low_memory=False)
            with autocast(enabled=True):
              net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)

            disp = disp + delta_disp.float()
            if itr < self.valid_iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)
        return disp_up




if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)
    ckpt_dir = "/offboard/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth"
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    disp = torch.randn(1, 1, 96, 160).cuda().float()
    onnx_path = "/offboard/FoundationStereo/assets/simplified.onnx"
    with torch.no_grad(): 
        model = FoundationStereoOnnx(args, 22)
        model.cuda()
        model.eval()
        out = model(disp)
        import pdb; pdb.set_trace()
        # Run optimization passes to reduce model size
        torch.onnx.export(
            model,
            (disp),
            onnx_path,
            opset_version=17,
            input_names = ['disp'],
            output_names = ['disp_out'],
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            export_params=True,
        )
        
        # Optimize the ONNX model to reduce size
        print("Optimizing ONNX model...")
        
        onnx_model = onnx.load(onnx_path)

    ######################### onnx shape inference
    onnx_path_inferred = "/offboard/FoundationStereo/assets/simplified_inferred.onnx"
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print('Model was successfully converted to ONNX format.')
    inferred_model_onnx = SymbolicShapeInference.infer_shapes(
        model_onnx, auto_merge=False, verbose=True)
    onnx.save(inferred_model_onnx, onnx_path_inferred)
    logging.info(f"ONNX model saved to {onnx_path_inferred}")
    ################# TRT Export ##################
    trt_cache_path = "/offboard/FoundationStereo/assets/trt_cache"
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,                     # Select GPU to execute
            "trt_engine_cache_enable": True,
            'trt_max_workspace_size': 4294967296,
            # 'trt_engine_cache_path': f"{trt_cache_path}/{trt_engine_name}",
            'trt_engine_cache_path': trt_cache_path,
            'trt_fp16_enable': True,              # Enable FP16 precision for faster inference              
            # 'trt_layer_norm_fp32_fallback': True, 
        }),
    ]
    sess_opt = ort.SessionOptions()
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opt.log_severity_level = 0
    # onnx_path_new = "/offboard/FoundationStereo/pretrained_models/onnx/foundation_stereo_23-51-11.onnx"
    sess = ort.InferenceSession(onnx_path_inferred, providers=providers, sess_options=sess_opt)
