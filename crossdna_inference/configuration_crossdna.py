from transformers import PretrainedConfig

class CrossDNAConfig(PretrainedConfig):
    model_type = "crossdna"

    def __init__(
        self,
        
        alphabet_size=5,
        d_model=128,
        block_size=1024,
        depth=6,
        drop_path_rates=(0.0, 0.05),
        dropout=0.15,

        pretrain=True,
        for_representation=False,
        use_s_scan=True,
        use_bridge=True,
        use_mem=False,
        use_rc_kl=False,
        use_barlow=False,
        use_tv=False,

        sem_max_weight=0.12,
        sem_warmup_steps=10000,
        aux_ce_weight=0.0,
        gate_freeze_steps=5000,
        detach_gate=False,
        gate_sup_weight=0.02,
        gate_sup_warmup_steps=500,
        gate_temp=2.0,

        
        transformer_cfg=None,
        comba_cfg=None,

        
        pad_token_id=4,   # [PAD]
        bos_token_id=2,   # [BOS]
        sep_token_id=1,   # [SEP]
        cls_token_id=0,   # [CLS]
        mask_token_id=3,  # [MASK]
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )
        self.alphabet_size = alphabet_size
        self.d_model = d_model
        self.block_size = block_size
        self.depth = depth
        self.drop_path_rates = list(drop_path_rates) if drop_path_rates is not None else None
        self.dropout = dropout

        self.pretrain = pretrain
        self.for_representation = for_representation
        self.use_s_scan = use_s_scan
        self.use_bridge = use_bridge
        self.use_mem = use_mem
        self.use_rc_kl = use_rc_kl
        self.use_barlow = use_barlow
        self.use_tv = use_tv

        self.sem_max_weight = sem_max_weight
        self.sem_warmup_steps = sem_warmup_steps
        self.aux_ce_weight = aux_ce_weight
        self.gate_freeze_steps = gate_freeze_steps
        self.detach_gate = detach_gate
        self.gate_sup_weight = gate_sup_weight
        self.gate_sup_warmup_steps = gate_sup_warmup_steps
        self.gate_temp = gate_temp

        self.transformer_cfg = transformer_cfg or {
            "hidden_size": d_model,
            "norm_eps": 1e-5,
            "max_position_embeddings": 1024,  
            "hidden_ratio": 4.0,
            "hidden_act": "swish",
            "fuse_swiglu": True,
            "attn": {
                "num_heads": 8,
                "num_kv_heads": 8,
                "qkv_bias": False,
                "window_size": 2048,
                "rope_theta": 10000
            }
        }
        self.comba_cfg = comba_cfg or {
            "hidden_size": d_model,
            "expand_v": 1,
            "head_dim": 64,
            "num_heads": 8,
            "use_gate": True,
            "mode": "chunk",
            "use_short_conv": True,
            "correction_factor": 0.02,
            "conv_size": 4,
            "norm_eps": 1e-5,
        }

        # 方便AutoModel推断 vocab_size
        self.vocab_size = self.alphabet_size