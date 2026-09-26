"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn

from vshavc.colormnet2.model.aggregate import aggregate
from vshavc.colormnet2.model.modules import *
from vshavc.colormnet2.model.memory_util import *

from vshavc.colormnet2.model.attention import LocalGatedPropagation
from vshavc.colormnet2.colormnet2_logbuffer import log_warning as _buf_warning


def _warn(msg, *args):
    """Queue a load_weights warning into the shared CMNET2 log buffer.

    The buffer is the single log channel of the model build: the RPC client
    drains it and forwards the messages to the VapourSynth log (remote,
    encode_mode=0), while the render flushes it right after the build when
    running locally.  Python logging is deliberately not used here: vsscript
    hosts (vspipe, vsViewer) cannot route logging from a non-main thread -
    in remote mode the model build runs inside the RPC server thread - and
    the VS logging bridge would raise AttributeError
    ('PythonVSScriptLoggingBridge' object has no attribute 'parent'),
    aborting the server-side initialize.
    """
    _buf_warning((msg % args) if args else str(msg))


class ColorMNet(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        # print(f'Single object mode: {self.single_object}')

        self.backbone = config.get('backbone', 'dinov3')
        self.key_encoder = KeyEncoder_DINOv2_v6(backbone=self.backbone,
                                                 dinov3_weights_dir=config.get('dinov3_weights_dir'))

        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)

        # Projection from f16 feature space to key/value space
        self.key_proj = KeyProjection(1024, self.key_dim) # 1024 -> 384 -> 3072

        self.short_term_attn = LocalGatedPropagation(d_qk=64, # 256
                                          d_vu=512 * 2,
                                          num_head=1,
                                          dilation=1,
                                          use_linear=False,
                                          dropout=0,
                                          d_att=64, # 128
                                          max_dis=7,
                                          expand_ratio=1)

        self.decoder = Decoder(self.value_dim, self.hidden_dim)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def encode_key(self, frame, need_sk=True, need_ek=True): 
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        f16, f8, f4 = self.key_encoder(frame)
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g16, h16

    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key, 
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def read_memory_short(self, query_key, memory_key, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        size_2d = query_key.shape[-2:]
        memory_value_short, _ = self.short_term_attn(query_key, memory_key, memory_value, None, size_2d)

        memory_value_short = memory_value_short.permute(1, 2, 0).view(batch_size, num_objects, self.value_dim, *memory_value.shape[-2:])

        return memory_value_short
    
    def segment(self, multi_scale_features, memory_readout,
                    hidden_state, selector=None, h_out=True, strip_bg=True): 

        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        
        prob = torch.tanh(logits)
        logits = prob

        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'read_memory_short':
            return self.read_memory_short(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0]//3
            # print(f'Hyperparameters read from the model weights: 'f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            if 'key_dim' not in config:
                self.key_dim = 64
                # print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                # print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                # print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    # print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        # print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    # else:
                    #   print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        if getattr(self, 'backbone', 'dinov3') == 'dinov3':
            # Shape-compatibility filter (no longer a blanket drop by prefix).
            # network2 can contain mutually incompatible structures (Segmentor DINOv2
            # vs Segmentor_DINOv3: different names, but also 2 keys with the SAME
            # name and different shape, e.g. 'backbone.norm.weight'/'bias' 384 vs
            # 768). strict=False alone is not enough: PyTorch still raises a
            # RuntimeError on a shape mismatch for a key that exists in both
            # state_dicts, regardless of strict. So only what is truly incompatible
            # with the already-instantiated model is discarded, not the entire
            # 'key_encoder.network2.*' prefix regardless - this lets a genuine
            # dinov3 checkpoint (fused or trained) actually be reloaded when its
            # keys match.
            own_state = self.state_dict()
            NETWORK2_PREFIX = 'key_encoder.network2.'
            filtered = {}
            remapped = 0
            for k, v in src_dict.items():
                if not k.startswith(NETWORK2_PREFIX):
                    filtered[k] = v
                    continue
                # Legacy naming: the fused checkpoints exported by the upstream
                # training pipeline store the DINOv3 ViT layers as
                # 'backbone.model.layer.*', while the inference model (the
                # transformers 4.x naming, kept by the native implementation)
                # uses 'backbone.layer.*'. Same tensors, same shapes - normalize
                # the key so the weights actually load instead of being
                # silently discarded by the existence filter below.
                if k.startswith(NETWORK2_PREFIX + 'backbone.model.layer.'):
                    k = NETWORK2_PREFIX + 'backbone.layer.' + k[len(NETWORK2_PREFIX + 'backbone.model.layer.'):]
                    remapped += 1
                if k not in own_state:
                    _warn(
                        "load_weights (backbone=dinov3): key '%s' absent from the "
                        "current model, discarded (checkpoint shape=%s)", k, tuple(v.shape))
                    continue
                if tuple(v.shape) != tuple(own_state[k].shape):
                    _warn(
                        "load_weights (backbone=dinov3): key '%s' with incompatible "
                        "shape (checkpoint=%s, expected=%s), discarded",
                        k, tuple(v.shape), tuple(own_state[k].shape))
                    continue
                filtered[k] = v
            if remapped:
                _warn(
                    "load_weights (backbone=dinov3): %d legacy 'backbone.model.layer.*' "
                    "keys remapped to 'backbone.layer.*'", remapped)
            result = self.load_state_dict(filtered, strict=False)
            # strict=False (needed above for the legitimate network2 filter)
            # also acted as a silent safety net for a completely wrong file
            # (e.g. a raw training checkpoint - first-level keys
            # 'effective_step'/'model'/'optimizer'/'scaler'/'scheduler', none
            # of which matches a real parameter): no real weight gets loaded,
            # but no error/warning signals it, the model stays at random
            # initialization. Threshold: the only LEGITIMATE exclusion of the
            # network2 prefix (the DinoV3 backbone, ~86M/190M total parameters
            # - about 45%) is the highest expected "missing" case in a correct
            # load; 90% stays well above that margin (2x), discarding only the
            # near-total mismatch case (no real weight in common, typically
            # >99% "missing" for a wholly foreign file), without touching a
            # legitimate load even in the extreme case of only the backbone
            # being incompatible.
            MISSING_FRACTION_THRESHOLD = 0.90
            missing_fraction = len(result.missing_keys) / len(own_state)
            if missing_fraction > MISSING_FRACTION_THRESHOLD:
                raise RuntimeError(
                    f"load_weights: {len(result.missing_keys)}/{len(own_state)} "
                    f"model parameters ({missing_fraction:.0%}) were left "
                    f"unmatched in the provided file - the provided file looks like "
                    f"a raw training checkpoint (first-level keys like "
                    f"'effective_step'/'model'/'optimizer'/'scaler'/'scheduler'), not "
                    f"a weights-only file. Check the path, or export the weights "
                    f"with training/export_weights.py."
                )
        else:
            self.load_state_dict(src_dict)
