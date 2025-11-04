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
from Utils import vis_disparity, depth2xyzmap, toOpen3dCloud
import open3d as o3d
# Add numpy.core.multiarray.scalar to the safe globals list right at the start
add_safe_globals([scalar])

class FoundationStereoOnnx(nn.Module):
    def __init__(self, model, img_shape, valid_iters):
        super().__init__()
        self.model = model
        self.valid_iters = valid_iters
        # self.padder = InputPadder(img_shape, divis_by=32, force_square=False)

    @torch.no_grad()
    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        # left, right = self.padder.pad(left, right)
        with torch.amp.autocast('cuda', enabled=True):
            disp = self.model.forward_onnx(left, right, iters=self.valid_iters, test_mode=True)
        # return self.padder.unpad(disp)
        return disp


def save_outputs(disp, padder, img0_ori, output_path, engine_type):
    intrinsic_file = "/offboard/FoundationStereo/assets/K_zed.txt"
    disp = padder.unpad(disp.float())
    disp = disp.float()
    disp = disp.data.cpu().numpy().reshape(360,640)
    vis = vis_disparity(disp)
    imageio.imwrite(f'{output_path}/vis_{engine_type}.png', vis)
    with open(intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    depth = K[0,0]*baseline/disp
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(360*640, 3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=10.0)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{output_path}/cloud_{engine_type}.ply', pcd)
    # logging.info("Visualizing point cloud. Press ESC to exit.")
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.get_render_option().point_size = 1.0
    # vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    # vis.run()
    # vis.destroy_window()

if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)

    ckpt_dir = "/offboard/FoundationStereo/pretrained_models/11-33-40/model_best_bp2.pth"
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    
    # left_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    # right_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    valid_iters = 2
    output_path = "/offboard/FoundationStereo/output"
    onnx_path = "/offboard/FoundationStereo/assets/foundation_stereo.onnx"
    onnx_path_inferred = "/offboard/FoundationStereo/assets/foundation_stereo_inferred.onnx"
    left_img = imageio.imread("/offboard/FoundationStereo/assets/zed_left.png")[:,:,:3]
    img0_ori = left_img.copy()
    left_img = torch.as_tensor(left_img).cuda().float()[None].permute(0,3,1,2)
    input_padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
    right_img = imageio.imread("/offboard/FoundationStereo/assets/zed_right.png")[:,:,:3]
    right_img = torch.as_tensor(right_img).cuda().float()[None].permute(0,3,1,2)
    left_img, right_img = input_padder.pad(left_img, right_img)
    # height = 480
    # width = 640
    # left_img = torch.randn(1, 3, height, width).cuda().float()
    # right_img = torch.randn(1, 3, height, width).cuda().float()
    
    
    with torch.no_grad():
        foundation_model = FoundationStereo(args)
        ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
        logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        foundation_model.load_state_dict(ckpt['model'])
        foundation_model.cuda()
        foundation_model.eval()    
        model = FoundationStereoOnnx(foundation_model, left_img.shape, valid_iters)
        model.cuda()
        model.eval()
        out = model(left_img, right_img)
        save_outputs(out, input_padder, img0_ori, output_path, "torch")
        # Run optimization passes to reduce model size
        torch.onnx.export(
            model,
            (left_img, right_img),
            onnx_path,
            opset_version=17,
            input_names = ['left', 'right'],
            output_names = ['disp'],
            # external_data_format=True,
            # export_params=True,
            # dynamic_axes={
            #     "left": { 2: "height", 3: "width"},
            #     "right": {2: "height", 3: "width"},
            #     "disp": { 2: "height", 3: "width"}
            # },
            # dynamic_axes={
            #     'left': {0 : 'batch_size'},
            #     'right': {0 : 'batch_size'},
            #     'disp': {0 : 'batch_size'}
            # },
            # dynamic_axes=None,
            # training=torch.onnx.TrainingMode.EVAL,
            # do_constant_folding=True,
            # export_params=True,
        )
        
        # Optimize the ONNX model to reduce size
        print("Optimizing ONNX model...")
        
        onnx_model = onnx.load(onnx_path)

    ######################### onnx shape inference
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print('Model was successfully converted to ONNX format.')
    inferred_model_onnx = SymbolicShapeInference.infer_shapes(
        model_onnx, auto_merge=False, verbose=True)
    onnx.save(inferred_model_onnx, onnx_path_inferred)
    logging.info(f"ONNX model saved to {onnx_path_inferred}")
    # import pdb; pdb.set_trace()
    ################# TRT Export ##################
    trt_cache_path = "/offboard/FoundationStereo/assets/trt_cache"
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,                     # Select GPU to execute
            "trt_engine_cache_enable": True,
            'trt_max_workspace_size': 8589934592,
            # 'trt_engine_cache_path': f"{trt_cache_path}/{trt_engine_name}",
            'trt_engine_cache_path': trt_cache_path,
            'trt_engine_cache_prefix': "foundation_stereo",
            'trt_fp16_enable': True,              # Enable FP16 precision for faster inference              
            # 'trt_layer_norm_fp32_fallback': True, 
        }),
    ]
    sess_opt = ort.SessionOptions()
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opt.log_severity_level = 1
    # onnx_path_new = "/offboard/FoundationStereo/pretrained_models/onnx/foundation_stereo_23-51-11.onnx"
    sess = ort.InferenceSession(onnx_path_inferred, providers=providers, sess_options=sess_opt)
    io_binding = sess.io_binding()
    device_type = right_img.device.type
    binded_disp = torch.zeros_like(out)
    # Reinstall onnxruntime-gpu if encountering device type related error
    io_binding.bind_input(
        name='left',
        device_type=device_type,
        device_id=0,
        element_type=np.float32,
        shape=left_img.shape,
        buffer_ptr=left_img.data_ptr(),
    )
    io_binding.bind_input(
        name='right',
        device_type=device_type,
        device_id=0,
        element_type=np.float32,
        shape=right_img.shape,
        buffer_ptr=right_img.data_ptr(),
    )
    io_binding.bind_output(
        name='disp',
        device_type=device_type,
        device_id=0,
        element_type=np.float32,
        shape=binded_disp.shape,
        buffer_ptr=binded_disp.data_ptr(),
    )
    # sess.run_with_iobinding(io_binding)
    with torch.no_grad():
        timer_pytorch = Timer(
            # The computation which will be run in a loop and timed.
            stmt="model(left_img, right_img)",
            setup="""
            """,
            
            globals={
                "model": model,
                "left_img": left_img,
                "right_img": right_img
            },
            # Control the number of threads that PyTorch uses. (Default: 1)
            num_threads=1,
        )
        timer_onnx = Timer(
            # The computation which will be run in a loop and timed.
            stmt="sess.run_with_iobinding(io_binding)",
            # `setup` will be run before calling the measurement loop, and is used to
            # populate any state which is needed by `stmt`
            setup="""
            """,
            
            globals={
                "sess": sess,
                "io_binding": io_binding,
            },
            # Control the number of threads that PyTorch uses. (Default: 1)
            num_threads=1,
        )
        m_torch: Measurement = timer_pytorch.blocked_autorange(min_run_time=1)
        m_onnx: Measurement = timer_onnx.blocked_autorange(min_run_time=1)
    print(m_torch)
    print(m_onnx)
    save_outputs(binded_disp, input_padder, img0_ori, output_path, "trt")