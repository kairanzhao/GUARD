import torch
import math
# import lpips
from transformers import CLIPProcessor, CLIPModel

from diffusers import StableDiffusionPipeline, DDPMScheduler
# from diffusers.utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from optim_utils import measure_CLIP_similarity
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import AdamW
from typing import Optional, Dict, Any, List
import types
import json, copy


def find_pre_eot_and_eot(token_ids, tokenizer):
    """
    Args:
        token_ids: list[int] (single sequence) or 1D torch.Tensor on CPU
        tokenizer: has .eos_token_id and .pad_token_id (may be None)

    Returns:
        (pre_eot_idx, eot_idx)
        pre_eot_idx: index of the last *content* token (non-EOS, non-PAD). None if not found.
        eot_idx: index of the first EOS or PAD that follows pre_eot_idx; if none, fallback to last index.
    """
    if token_ids is None:
        return None, None
    if hasattr(token_ids, "detach"):  # torch.Tensor
        token_ids = token_ids.detach().cpu().tolist()

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    L = len(token_ids)
    if L == 0:
        return None, None

    pre_eot_idx = None
    for j in range(L - 1, -1, -1):
        tid = token_ids[j]
        is_eos = (eos_id is not None and tid == eos_id)
        is_pad = (pad_id is not None and tid == pad_id)
        if not (is_eos or is_pad):
            pre_eot_idx = j
            break

    if pre_eot_idx is None:
        pre_eot_idx = max(0, L - 2)

    eot_idx = None
    if eos_id is not None:
        for k in range(pre_eot_idx + 1, L):
            if token_ids[k] == eos_id:
                eot_idx = k
                break
    if eot_idx is None and pad_id is not None:
        for k in range(pre_eot_idx + 1, L):
            if token_ids[k] == pad_id:
                eot_idx = k
                break
    if eot_idx is None:
        eot_idx = L - 1

    return pre_eot_idx, eot_idx


def torch_cos_sim(v, cos_theta, n_vectors=1, EXACT=True):
    """
    EXACT - if True, all vectors will have exactly cos_theta similarity.
            if False, all vectors will have >= cos_theta similarity
    v - original vector (1D tensor)
    cos_theta -cos similarity in range [-1,1]
    """
    u = v / torch.norm(v)
    u = u.unsqueeze(0).repeat(n_vectors, 1)

    r = (torch.rand([n_vectors, len(v)]) * 2 - 1).to(v.device).to(v.dtype)

    uperp = torch.stack([r[i] - (torch.dot(r[i], u[i]) * u[i]) for i in range(len(u))])
    uperp = uperp / (torch.norm(uperp, dim=1).unsqueeze(1).repeat(1, v.shape[0]))

    if not EXACT:
        cos_theta = torch.rand(n_vectors) * (1 - cos_theta) + cos_theta
        cos_theta = cos_theta.unsqueeze(1).repeat(1, v.shape[0])

    w = cos_theta * u + torch.sqrt(1 - torch.tensor(cos_theta) ** 2) * uperp

    return w

class _XAttnLogger:
    def __init__(self, tokenizer, trigger_token_ids: List[int], tail_token_ids: List[int],
                 gate_suffix: float = 1.0, gate_tail: float = 1.0,
                 gate_window: str = "late", gate_frac: float = 0.25, enable_loaded_mass: bool = False,
                 bias_schedule: str = "constant", 
                 bias_start_frac: float = 0.0,  
                 bias_end_frac: float = 1.0,  
                 intervention_blocks: Optional[List[str]] = None,
                 bias_on: str = "cond",
                 pst_hot_map: Optional[Dict[str, List[int]]] = None,
                 eot_hot_map: Optional[Dict[str, List[int]]] = None,
                 bot_hot_map: Optional[Dict[str, List[int]]] = None,
                 dynamic_head_selection: bool = False,
                 dynamic_scout_steps: int = 5,
                 dynamic_top_k_per_block: float = 3.0,
                 dynamic_top_k_global: float = 5.0,
                 dynamic_metric: str = "eot_mass_peakiness", 
                 dynamic_min_threshold: float = 0.0,
                 trace_per_block: bool = True,
                 basin_beta: float = 0.0,
                 basin_scout_blocks: Optional[List[str]] = None, 
                 spike_threshold: float = 0.0,   
                 spike_penalty: float = 0.0,    
                 spike_scale: float = 1.0,      
                 log_stats: bool = True,
                 ):
        self.tokenizer = tokenizer
        self.trigger_ids = set(trigger_token_ids or []) 
        self.tail_ids = set(tail_token_ids or [])
        self.gate_suffix = float(gate_suffix)
        self.gate_tail = float(gate_tail)
        self.gate_window = gate_window
        self.gate_frac = float(gate_frac)
        self.enable_loaded_mass = enable_loaded_mass
        self.bias_schedule = str(bias_schedule)
        self.bias_start_frac = float(bias_start_frac)
        self.bias_end_frac = float(bias_end_frac)
        self.intervention_blocks = set(intervention_blocks) if intervention_blocks else None
        self.bias_on = str(bias_on)

        self.pst_hot_map = pst_hot_map or {}
        self.eot_hot_map = eot_hot_map or {}
        self.bot_hot_map = bot_hot_map or {}
        
        self.dynamic_head_selection = bool(dynamic_head_selection)
        self.dynamic_scout_steps = int(dynamic_scout_steps)
        self.dynamic_top_k_per_block = float(dynamic_top_k_per_block)
        self.dynamic_top_k_global = float(dynamic_top_k_global)
        self.dynamic_metric = str(dynamic_metric)
        self.dynamic_min_threshold = float(dynamic_min_threshold)
        self.trace_per_block = bool(trace_per_block)
        self.stats_per_block = {} 
        self._static_pst_hot_map = self.pst_hot_map
        self._static_eot_hot_map = self.eot_hot_map
        self._static_bot_hot_map = self.bot_hot_map
        self._scout_stats = {} 
        self._scout_finalized = not self.dynamic_head_selection 
        self.stats = {"TailMass": [], "TriggerMass": [], "Entropy": []}
        if self.enable_loaded_mass:
            self.stats["LoadedMass"] = []

        self._attached = False
        self._orig_forwards = {}
        self._curr_step = 0
        self._num_steps = 1
        self._global_batch_token_ids = None 
        self._global_pre_eot_index = None  
        self._global_eot_index = None     

        self._indices_by_label = {}    
        self._stats_by_label = {}
        self._accum_by_label = {}

        self.runs: list = []        
        self._run_counter: int = 0  

        self._pre_eot_index = None  
        self._eot_index = None    

        self.basin_beta = basin_beta
        self.basin_scout_blocks = basin_scout_blocks 
        self.basin_ref_mass = 0.0 
        self.basin_history = []  
        self.in_basin = True    
        self.row_slices = {}  
        self._batch_comp = None

        self.spike_threshold = float(spike_threshold)
        self.spike_penalty = float(spike_penalty)
        self.spike_scale = float(spike_scale)

        self.log_stats = bool(log_stats)


    def _run_scout_analysis(self, marginal_per_head: torch.Tensor, attn_probs_head: torch.Tensor, block_name: str):
        """
        Vectorized accumulation of per-head stats.
        marginal_per_head: [H, K]
        attn_probs_head: [B, H, Q, K]
        """
        if block_name not in self._scout_stats:
            H = marginal_per_head.shape[0]
            device = marginal_per_head.device
            self._scout_stats[block_name] = {
                "eot_mass": torch.zeros(H, device=device),
                "pst_mass": torch.zeros(H, device=device),
                "bot_mass": torch.zeros(H, device=device),
                "content_mass": torch.zeros(H, device=device),
                "peakiness": torch.zeros(H, device=device),
                "n": 0
            }
        
        stats = self._scout_stats[block_name]
        H, K = marginal_per_head.shape
        
        p = attn_probs_head.clamp_min(1e-12)
        ent_per_head = (-p * p.log()).sum(dim=-1).mean(dim=0).mean(dim=1) 
        stats["peakiness"] += 1.0 / (ent_per_head + 1e-6)
        
        idx_bot = 0
        idx_pst = self._global_pre_eot_index - 1 if self._global_pre_eot_index is not None and self._global_pre_eot_index > 0 else -1
        idx_eot = self._global_eot_index if self._global_eot_index is not None else -1

        if 0 <= idx_bot < K:
            stats["bot_mass"] += marginal_per_head[:, idx_bot]
        if 0 <= idx_pst < K:
            stats["pst_mass"] += marginal_per_head[:, idx_pst]
        if 0 <= idx_eot < K:
            stats["eot_mass"] += marginal_per_head[:, idx_eot]
            
        if idx_pst > 0:

            stats["content_mass"] += marginal_per_head[:, 1:idx_pst+1].sum(dim=1)
            
        stats["n"] += 1

    
    def _finalize_scout_analysis(self):
        """
        Computes final "hotness" scores and populates the self.*_hot_map dicts.
        Logic:
          1. If dynamic_top_k_global > 0: Select top K heads across the entire U-Net.
          2. Else: Select top K (or top %) heads PER BLOCK using dynamic_top_k_per_block.
        """
        print(f"[XAttnLogger] Finalizing dynamic hot head analysis...")
        
        new_pst_map = {b: [] for b in self._scout_stats.keys()}
        new_eot_map = {b: [] for b in self._scout_stats.keys()}
        new_bot_map = {b: [] for b in self._scout_stats.keys()}

        full_head_scores = {"pst": {}, "eot": {}, "bot": {}} 

        block_scores_eot = {}
        block_scores_pst = {}
        block_scores_bot = {}
        
        global_scores_eot = []
        global_scores_pst = []
        global_scores_bot = []

        for block_name, block_stats in self._scout_stats.items():
            full_head_scores["pst"][block_name] = {}
            full_head_scores["eot"][block_name] = {}
            full_head_scores["bot"][block_name] = {}
            
            block_scores_eot[block_name] = []
            block_scores_pst[block_name] = []
            block_scores_bot[block_name] = []

            n = max(1, block_stats["n"])
            H = block_stats["eot_mass"].shape[0]

            for h in range(H):
                avg_eot_mass = block_stats["eot_mass"][h].item() / n
                avg_pst_mass = block_stats["pst_mass"][h].item() / n
                avg_bot_mass = block_stats["bot_mass"][h].item() / n
                
                if self.dynamic_metric == "eot_vs_content": 
                    avg_content_mass = block_stats["content_mass"][h].item() / n
                    denom = avg_content_mass + 1e-6
                    score_eot = avg_eot_mass / denom
                    score_pst = avg_pst_mass / denom
                    score_bot = avg_bot_mass / denom
                else: 
                    avg_peakiness = block_stats["peakiness"][h].item() / n
                    score_eot = avg_eot_mass * avg_peakiness
                    score_pst = avg_pst_mass * avg_peakiness
                    score_bot = avg_bot_mass * avg_peakiness

                full_head_scores["pst"][block_name][h] = score_pst
                full_head_scores["eot"][block_name][h] = score_eot
                full_head_scores["bot"][block_name][h] = score_bot
                
                global_scores_eot.append((score_eot, block_name, h))
                global_scores_pst.append((score_pst, block_name, h))
                global_scores_bot.append((score_bot, block_name, h))
                block_scores_eot[block_name].append((score_eot, h))
                block_scores_pst[block_name].append((score_pst, h))
                block_scores_bot[block_name].append((score_bot, h))

        threshold = max(self.dynamic_min_threshold, 0.0)
        
        if self.dynamic_top_k_global > 0.0:
            print(f"[XAttnLogger] Mode: GLOBAL Selection (Top {self.dynamic_top_k_global})")
            
            raw_k = self.dynamic_top_k_global
            total_heads = len(global_scores_eot)
            if 0.0 < raw_k < 1.0:
                k = max(1, int(total_heads * raw_k))
            else:
                k = int(raw_k)
            
            sorted_eot = sorted(global_scores_eot, reverse=True)
            sorted_pst = sorted(global_scores_pst, reverse=True)
            sorted_bot = sorted(global_scores_bot, reverse=True)
            
            for score, b, h in sorted_eot[:k]:
                if score > threshold: new_eot_map[b].append(h)
            for score, b, h in sorted_pst[:k]:
                if score > threshold: new_pst_map[b].append(h)
            for score, b, h in sorted_bot[:k]:
                if score > threshold: new_bot_map[b].append(h)

        elif self.dynamic_top_k_per_block > 0.0:
            print(f"[XAttnLogger] Mode: PER-BLOCK Selection (Top {self.dynamic_top_k_per_block})")
            raw_k = self.dynamic_top_k_per_block
            
            for b_name in self._scout_stats.keys():
                scores = block_scores_eot[b_name]
                H_block = len(scores)
                if 0.0 < raw_k <= 1.0:
                    k = max(1, int(H_block * raw_k))
                else:
                    k = int(raw_k)
                
                scores.sort(key=lambda x: x[0], reverse=True)
                
                for score, h in scores[:k]:
                    if score > threshold: new_eot_map[b_name].append(h)

                scores_pst = block_scores_pst[b_name]
                scores_pst.sort(key=lambda x: x[0], reverse=True)
                for score, h in scores_pst[:k]:
                    if score > threshold: new_pst_map[b_name].append(h)

                scores_bot = block_scores_bot[b_name]
                scores_bot.sort(key=lambda x: x[0], reverse=True)
                for score, h in scores_bot[:k]:
                    if score > threshold: new_bot_map[b_name].append(h)
        

        self.pst_hot_map = new_pst_map
        self.eot_hot_map = new_eot_map
        self.bot_hot_map = new_bot_map
        
        self.runs.append({
            "label": "hot_head_map",
            "metric": self.dynamic_metric,
            "pst_hot_map": new_pst_map,
            "eot_hot_map": new_eot_map,
            "bot_hot_map": new_bot_map,
            "full_scores": full_head_scores 
        })
        
        print(f"[XAttnLogger] => Selected EOT Hot Map: {new_eot_map}")
        
        self._scout_finalized = True

    def _get_bias_scaler(self) -> float:
        """
        Calculates the current bias scaler (0.0 to 1.0) based on the schedule.
        """

        S = max(1, self._num_steps)
        s_frac = max(0.0, min(1.0, self.bias_start_frac))
        e_frac = max(0.0, min(1.0, self.bias_end_frac))
        if e_frac < s_frac:
            s_frac, e_frac = e_frac, s_frac  

        start_step = int(s_frac * S)
        end_step   = int(e_frac * S)

        if self._curr_step < start_step or self._curr_step >= end_step:
            return 0.0 
        
        span = max(1, end_step - start_step)
        p = (self._curr_step - start_step) / float(span)
        
        if self.bias_schedule == "linear_fadein":
            s = p
        elif self.bias_schedule == "linear_fadeout":
            s = 1.0 - p
        elif self.bias_schedule == "cosine_fadein":
            s = 0.5 * (1.0 - math.cos(math.pi * p))
        elif self.bias_schedule == "cosine_fadeout":
            s = 0.5 * (1.0 + math.cos(math.pi * p))
        else: 
            s = 1.0
            
        return float(s)


    def _update_basin_state(self, step_idx: int):
        """
        Calculates A_t (avg EOT mass in early blocks), updates A_ref, 
        and determines if we are still in the basin.
        """
        if not self.stats_per_block:
            return

        target_blocks = []
        for bname in self.stats_per_block.keys():
            is_target = False
            if self.basin_scout_blocks is None:
                if bname.startswith("down_blocks") or bname.startswith("mid_block"):
                    is_target = True
            else:
                if bname in self.basin_scout_blocks:
                    is_target = True
            
            if is_target:
                target_blocks.append(bname)

        if not target_blocks:
            return

        current_masses = []
        for bname in target_blocks:
            eot_series = self.stats_per_block[bname].get("EOTMass", [])
            if eot_series:
                current_masses.append(eot_series[-1])
        
        if not current_masses:
            return

        A_t = sum(current_masses) / len(current_masses)
        self.basin_history.append(A_t)

        if step_idx < 5:
            self.basin_ref_mass = max(self.basin_ref_mass, A_t)
        
        ref = max(self.basin_ref_mass, 1e-6)
        ratio = A_t / ref

        is_stuck = ratio > self.basin_beta

        self.in_basin = is_stuck
        if not is_stuck:
            pass


    def start_run(self, label: str = ""):
        if not label:
            self._run_counter += 1
            label = f"run_{self._run_counter:04d}"
        self._run_label = label # Used by global end_run()

        self.pst_hot_map = self._static_pst_hot_map
        self.eot_hot_map = self._static_eot_hot_map
        self.bot_hot_map = self._static_bot_hot_map
        
        self._scout_stats = {}
        if self.dynamic_head_selection:

            self._scout_finalized = False
            self.pst_hot_map = {}
            self.eot_hot_map = {}
            self.bot_hot_map = {}
        else:
            self._scout_finalized = True

        self.stats = {
            "TailMass": [], "TriggerMass": [], "Entropy": [],
            "PreEOTMass": [], "EOTMass": [], "TokenMass": [],
        }
        self.stats_per_block = {}

        if self.enable_loaded_mass:
            self.stats["LoadedMass"] = []
        
        self._stats_by_label = {}
        self._accum_by_label = {}
        
        self.stats_by_label = {} 
        self._accum = {
            "TailMass": 0.0, "TriggerMass": 0.0, "Entropy": 0.0,
            "PreEOTMass": 0.0, "EOTMass": 0.0, "TokenMass": None, "n": 0
        }
        self.basin_ref_mass = 0.0
        self.basin_history = []
        self.in_basin = True

    def end_run(self, label: Optional[str] = None):
        """
        Finalizes and freezes a run.
        - If label is provided, freezes the stream from _stats_by_label[label].
        - If label is None, freezes the global self.stats.
        """
        stats_to_freeze = None
        run_label = label
        eot_idx_to_save = self._global_eot_index 
        pre_eot_idx_to_save = self._global_pre_eot_index 
        
        if label:

            if label in self._stats_by_label:
                stats_to_freeze = self._stats_by_label[label]
                run_label = label
                indices = self._indices_by_label.get(label, (None, None, None))
                pre_eot_idx_to_save = indices[1] 
                eot_idx_to_save = indices[2]  
            else:
                return
        else:
            self._flush_step()
            stats_to_freeze = self.stats
            run_label = self._run_label 
        
        frozen = {"label": run_label}
        for k, v in stats_to_freeze.items():
            frozen[k] = copy.deepcopy(v)
        
        if self.trace_per_block: 
            frozen["stats_per_block"] = copy.deepcopy(self.stats_per_block)
        
        frozen["pre_eot_idx"] = pre_eot_idx_to_save
        frozen["eot_idx"] = eot_idx_to_save
        self.runs.append(frozen)

    def begin_step(self):
        self._accum = {
            "TailMass": 0.0,
            "TriggerMass": 0.0,
            "Entropy": 0.0,
            "PreEOTMass": 0.0,
            "EOTMass": 0.0,
            "TokenMass": None,
            "n": 0,
        }

    def _accumulate_from_probs(self, attn_probs: torch.Tensor):
        km = self._compute_token_marginals(attn_probs)
        self._accum["TailMass"]    += km["TailMass"]
        self._accum["TriggerMass"] += km["TriggerMass"]
        self._accum["Entropy"]     += km["Entropy"]
        self._accum["PreEOTMass"]  += km["PreEOTMass"]
        self._accum["EOTMass"]     += km["EOTMass"]
        self._accum["n"]           += 1
        tv = km["TokenVec"] 
        if self._accum["TokenMass"] is None:
            self._accum["TokenMass"] = tv.detach().clone()
        else:
            self._accum["TokenMass"] += tv

    def _flush_step(self):
        n = max(1, self._accum["n"])
        self.stats["TailMass"].append(self._accum["TailMass"] / n)
        self.stats["TriggerMass"].append(self._accum["TriggerMass"] / n)
        self.stats["Entropy"].append(self._accum["Entropy"] / n)
        self.stats["PreEOTMass"].append(self._accum["PreEOTMass"] / n)
        self.stats["EOTMass"].append(self._accum["EOTMass"] / n)
        if self.enable_loaded_mass:
            self.stats["LoadedMass"].append(0.0)
        if self._accum["TokenMass"] is None:
            pass
        else:
            step_vec = (self._accum["TokenMass"] / n).detach().cpu().tolist()  # [K]
            self.stats["TokenMass"].append(step_vec)

        self._accum = {
            "TailMass": 0.0, "TriggerMass": 0.0, "Entropy": 0.0,
            "PreEOTMass": 0.0, "EOTMass": 0.0, "TokenMass": None, "n": 0
        }

    def set_schedule(self, step_idx: int, num_steps: int):
        self._curr_step = step_idx
        self._num_steps = max(1, num_steps)

    def set_batch_token_ids(self, token_ids: List[int]):
        self._global_batch_token_ids = token_ids

    def set_special_indices(self, pre_eot: Optional[int], eot: Optional[int]):
        """
        Set GLOBAL pre_eot and eot for __call__
        """
        self._global_pre_eot_index = pre_eot
        self._global_eot_index = eot

    def set_stream_indices(self, label: str, token_ids: List[int], pre_eot: Optional[int], eot: Optional[int]):
        """
        Stores the token_ids and EOT indices for a specific stream label.
        """
        if label:
            self._indices_by_label[label] = (token_ids, pre_eot, eot)

    def set_row_slices(self, row_slices: Dict[str, tuple]):
        """
        row_slices maps stream labels to [start, end) row indices in the [B*H, Q, K] tensor.
        Example: {"positive": (p0,p1), "negative": (n0,n1)}
        """
        self.row_slices = row_slices
        self._accum_by_label = {}
        self._stats_by_label = {}

    def set_batch_composition(self, N_uncond: int, N_pos: int, N_neg: int):
        """
        Sets the batch composition for stream-aware logging.
        N_uncond: Number of unconditional samples (e.g., N)
        N_pos:    Number of positive/neighbor samples (e.g., k*N)
        N_neg:    Number of negative/repulsive samples (e.g., N)
        """
        self._batch_comp = {"uncond": N_uncond, "pos": N_pos, "neg": N_neg}
        self.row_slices = {} 
    def _ensure_label_accum(self, label: str):
        if label not in self._accum_by_label:
            self._accum_by_label[label] = {
                "TailMass": 0.0, "TriggerMass": 0.0, "Entropy": 0.0,
                "PreEOTMass": 0.0, "EOTMass": 0.0, "TokenMass": None, "n": 0
            }
        if label not in self._stats_by_label:
            self._stats_by_label[label] = {
                "TailMass": [], "TriggerMass": [], "Entropy": [],
                "PreEOTMass": [], "EOTMass": [], "TokenMass": []
            }

    def _accumulate_from_probs_stream(self, attn_probs: torch.Tensor, label: str):
        """
        Same as _accumulate_from_probs, but into a per-label accumulator.
        """
        self._ensure_label_accum(label)
        km = self._compute_token_marginals(attn_probs, label=label) 
        
        acc = self._accum_by_label[label]
        acc["TailMass"]    += km["TailMass"]
        acc["TriggerMass"] += km["TriggerMass"]
        acc["Entropy"]     += km["Entropy"]
        acc["PreEOTMass"]  += km["PreEOTMass"]
        acc["EOTMass"]     += km["EOTMass"]
        acc["n"]           += 1
        tv = km["TokenVec"]
        if acc["TokenMass"] is None:
            acc["TokenMass"] = tv.detach().clone()
        else:
            acc["TokenMass"] += tv

    def _flush_step_stream(self, label: str):
        acc = self._accum_by_label.get(label)
        if not acc:
            return
        n = max(1, acc["n"])
        st = self._stats_by_label[label]
        st["TailMass"].append(acc["TailMass"] / n)
        st["TriggerMass"].append(acc["TriggerMass"] / n)
        st["Entropy"].append(acc["Entropy"] / n)
        st["PreEOTMass"].append(acc["PreEOTMass"] / n)
        st["EOTMass"].append(acc["EOTMass"] / n)
        if acc["TokenMass"] is not None:
            st["TokenMass"].append((acc["TokenMass"] / n).detach().cpu().tolist())
        # reset
        self._accum_by_label[label] = {
            "TailMass": 0.0, "TriggerMass": 0.0, "Entropy": 0.0,
            "PreEOTMass": 0.0, "EOTMass": 0.0, "TokenMass": None, "n": 0
        }



    def _in_gate_window(self) -> bool:
        if self.gate_window == "all":
            return True
        if self.gate_window == "late":
            return self._curr_step >= int((1.0 - self.gate_frac) * self._num_steps)
        return False

    def _gate_vector_for_tokens(self, token_ids: List[int]) -> Optional[torch.Tensor]:
        if token_ids is None or not self._in_gate_window():
            return None
        g = torch.ones(len(token_ids), dtype=torch.float32)
        for i, tid in enumerate(token_ids):
            if tid in self.tail_ids:
                g[i] *= self.gate_tail
            if tid in self.trigger_ids:
                g[i] *= self.gate_suffix
        return g
 
    def _compute_token_marginals(self, attn_probs: torch.Tensor, label: Optional[str] = None) -> Dict[str, float]:
        """
        attn_probs: [B*H, Q, K] (typical in diffusers cross-attn)
        We average over heads (H) and queries (Q), leaving a token-marginal over K.
        Then we sum over selected token indices for TailMass and TriggerMass.
        Entropy: average token-wise entropy over queries & heads.
        
        If 'label' is provided, uses stream-specific token_ids and indices.
        Otherwise, falls back to global token_ids and indices (for __call__).
        """
        
        batch_token_ids = self._global_batch_token_ids
        pre_eot_index = self._global_pre_eot_index  
        eot_index = self._global_eot_index
        
        if label and label in self._indices_by_label:
            token_ids, pre_eot, eot = self._indices_by_label[label]
            batch_token_ids = token_ids
            pre_eot_index = pre_eot
            eot_index = eot

        marginal = attn_probs.mean(dim=0).mean(dim=0) 
        token_vec = marginal
        p = attn_probs.clamp_min(1e-12)
        ent = (-p * p.log()).sum(dim=-1).mean().item()

        def _sum_for(ids):
            if not batch_token_ids or not ids:
                return 0.0
            idxs = [i for i, tid in enumerate(batch_token_ids) if tid in ids]
            if not idxs:
                return 0.0
            idxs = [i for i in idxs if i < marginal.shape[0]]
            if not idxs:
                return 0.0
            sel = marginal[idxs].sum().item()
            return float(sel)

        def _get_index_mass(idx: Optional[int]) -> float:
            if idx is None:
                return 0.0
            if idx < 0 or idx >= marginal.shape[0]:
                return 0.0
            return float(marginal[idx].item())

        return {
            "TailMass": _sum_for(self.tail_ids),
            "TriggerMass": _sum_for(self.trigger_ids),
            "Entropy": float(ent),
            "PreEOTMass": _get_index_mass(pre_eot_index),
            "EOTMass": _get_index_mass(eot_index),
            "TokenVec": token_vec,
        }
    
    def attach(self, unet):
        if self._attached:
            return
        for name, module in unet.named_modules():
            if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v") and getattr(module, "is_cross_attention", False):
                if name in self._orig_forwards:
                    continue

                self._orig_forwards[name] = module.forward
                def make_wrapped(mod, block_name=name):
                    orig = mod.forward
                    def wrapped(self_attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                        
                        is_intervention_block = self.intervention_blocks is None or block_name in self.intervention_blocks

                        if is_intervention_block and encoder_hidden_states is not None and self._global_batch_token_ids is not None:
                            gate_vec = self._gate_vector_for_tokens(self._global_batch_token_ids)
                            if gate_vec is not None and encoder_hidden_states.dim() == 3 and encoder_hidden_states.shape[1] == len(self._global_batch_token_ids):
                                scale = gate_vec.to(encoder_hidden_states.device).to(encoder_hidden_states.dtype)
                                encoder_hidden_states = encoder_hidden_states * scale.unsqueeze(0).unsqueeze(-1)

                        bsz, q_len, _ = hidden_states.shape

                        q = self_attn.to_q(hidden_states)
                        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                        k = self_attn.to_k(context)
                        v = self_attn.to_v(context)

                        def head_to_batch(x):
                            b, l, c = x.shape
                            x = x.view(b, l, self_attn.heads, -1)
                            x = x.permute(0, 2, 1, 3).reshape(b * self_attn.heads, l, -1)
                            return x
                        def batch_to_head(x, b, l):
                            x = x.view(b, self_attn.heads, l, -1)
                            x = x.permute(0, 2, 1, 3).reshape(b, l, -1)
                            return x

                        q = head_to_batch(q)
                        k = head_to_batch(k)
                        v = head_to_batch(v)

                        scale = 1.0 / math.sqrt(q.shape[-1])
                        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale

                        scaler = self._get_bias_scaler()
                        if is_intervention_block and self.spike_threshold > 0.0 and scaler > 0.0:
                            target_slices = [] 
                            H = self_attn.heads

                            if hasattr(self, "_batch_comp") and self._batch_comp:
                                N_u = self._batch_comp["uncond"]
                                N_p = self._batch_comp["pos"]
                                N_n = self._batch_comp["neg"]

                                start_p = N_u * H
                                end_p   = start_p + N_p * H
                                target_slices.append((start_p, end_p))

                            else:
                                total_rows = attn_scores.shape[0]
                                if total_rows % (2 * H) == 0:
                                    half = total_rows // 2
                                    target_slices.append((half, total_rows))
                                else:
                                    target_slices.append((0, total_rows))

                            for (slice_start, slice_end) in target_slices:
                                target_scores = attn_scores[slice_start:slice_end]
                                token_max_scores = target_scores.max(dim=1).values
                                content_scores = token_max_scores[:, 1:]
                            
                                mu = content_scores.mean(dim=-1, keepdim=True)
                                std = content_scores.std(dim=-1, keepdim=True)
                                z_scores = (token_max_scores - mu) / (std + 1e-6)
                                is_spike = z_scores > self.spike_threshold

                                is_spike[:, 0] = False
                                hot_heads = self.eot_hot_map.get(block_name)
                                
                                if hot_heads is not None:
                                    head_mask_1d = torch.zeros(H, device=is_spike.device, dtype=torch.bool)
                                    head_mask_1d[hot_heads] = True 
                                    
                                    slice_len = slice_end - slice_start
                                    num_batches = slice_len // H
                                    head_filter = head_mask_1d.repeat(num_batches).unsqueeze(1)
                                    
                                    is_spike = is_spike & head_filter
                                
                                if is_spike.any():
                                    spike_mask = is_spike.unsqueeze(1)
                                    current_slice = attn_scores[slice_start:slice_end]

                                    if self.spike_scale != 1.0:
                                        effective_scale = 1.0 + (self.spike_scale - 1.0) * scaler
                                        attn_scores[slice_start:slice_end] = torch.where(
                                            spike_mask, 
                                            current_slice * effective_scale,
                                            current_slice
                                        )
                                    elif self.spike_penalty != 0.0:
                                        effective_penalty = self.spike_penalty * scaler
                                        penalty_tensor = spike_mask.float() * effective_penalty
                                        attn_scores[slice_start:slice_end] = current_slice + penalty_tensor.to(attn_scores.dtype)

                        if is_intervention_block and self.dynamic_head_selection and not self._scout_finalized:
                            with torch.no_grad():
                                temp_attn_probs = torch.softmax(attn_scores, dim=-1) # [B*H, Q, K]
                                
                                H = self_attn.heads
                                K = temp_attn_probs.shape[-1]
                                
                                temp_attn_probs_head = temp_attn_probs.view(bsz, H, q_len, K)
                                temp_marginal_per_head = temp_attn_probs_head.mean(dim=0).mean(dim=1)
                                
                                if self._curr_step < self.dynamic_scout_steps:
                                    self._run_scout_analysis(temp_marginal_per_head, temp_attn_probs_head, block_name)
                                elif self._curr_step == self.dynamic_scout_steps:
                                    self._finalize_scout_analysis()
                        
                        if is_intervention_block and scaler > 0.0:
                            K = attn_scores.shape[-1]
                            H = self_attn.heads

                            head_bias_vec = attn_scores.new_zeros(H, 1, K)

                            pst_heads = self.pst_hot_map.get(block_name) 
                            eot_heads = self.eot_hot_map.get(block_name) 
                            bot_heads = self.bot_hot_map.get(block_name) 
                            
                            if self.bot_bias != 0.0:
                                bias_to_add = self.bot_bias * scaler
                                if bot_heads is None: 
                                    head_bias_vec[:, :, 0] += bias_to_add
                                else: 
                                    for h_idx in bot_heads:
                                        if 0 <= h_idx < H:
                                            head_bias_vec[h_idx, :, 0] += bias_to_add
                            
                            if torch.any(head_bias_vec != 0):
                                total_rows = attn_scores.shape[0] 
                                if self.bias_on == "cond":
                                    if hasattr(self, "_batch_comp") and self._batch_comp:
                                        N_u = self._batch_comp["uncond"]
                                        N_p = self._batch_comp["pos"]
                                        s = N_u * H
                                        e = s + N_p * H
                                        if e <= total_rows:
                                            if N_p > 1:
                                                bias_to_apply = head_bias_vec.repeat(N_p, 1, 1)
                                                attn_scores[s:e, :, :] += bias_to_apply
                                            else:
                                                attn_scores[s:e, :, :] += head_bias_vec
                                        else:
                                            attn_scores += head_bias_vec.repeat(total_rows // H, 1, 1)

                                    elif not hasattr(self, "_batch_comp") or not self._batch_comp:
                                        if total_rows == 2 * H: 
                                            attn_scores[H:, :, :] += head_bias_vec
                                        else:
                                            if total_rows == H:
                                                attn_scores[:H, :, :] += head_bias_vec
                                            else:
                                                attn_scores += head_bias_vec.repeat(total_rows // H, 1, 1)
                                    else:
                                        attn_scores += head_bias_vec.repeat(total_rows // H, 1, 1)
                                else:
                                    attn_scores += head_bias_vec.repeat(total_rows // H, 1, 1)
                        
                        
                        
                        if attention_mask is not None:
                            attn_scores = attn_scores + attention_mask

                        attn_probs = torch.softmax(attn_scores, dim=-1)  

                        if self.log_stats or self.trace_per_block:
                            with torch.no_grad():
                                H = self_attn.heads
                                
                                if self.trace_per_block:
                                    probs_to_log = None
                                    label_to_log = None

                                    if hasattr(self, "_batch_comp") and self._batch_comp:
                                        N_u = self._batch_comp.get("uncond", 0)
                                        N_p = self._batch_comp.get("pos", 0)
                                        
                                        idx_start = N_u * H
                                        idx_end   = idx_start + N_p * H
                                        
                                        if idx_end <= attn_probs.shape[0]:
                                            probs_to_log = attn_probs[idx_start:idx_end]
                                            label_to_log = "positive"
                                    
                                    elif (not hasattr(self, "_batch_comp") or not self._batch_comp):
                                        total_rows = attn_probs.shape[0]

                                        if total_rows % (2 * H) == 0 and total_rows >= 2 * H:
                                            N = total_rows // (2 * H)
                                            cond_start_index = N * H
                                            probs_to_log = attn_probs[cond_start_index:] 
                                            label_to_log = None 
                                        elif total_rows % H == 0:
                                            probs_to_log = attn_probs 
                                            label_to_log = None 
                                    
                                    if probs_to_log is not None and probs_to_log.shape[0] > 0:
                                        km = self._compute_token_marginals(probs_to_log, label=label_to_log)
                                        if block_name not in self.stats_per_block:
                                            self.stats_per_block[block_name] = {"EOTMass": [], "PreEOTMass": [], "Entropy": []}
                                        
                                        self.stats_per_block[block_name]["EOTMass"].append(km["EOTMass"])
                                        self.stats_per_block[block_name]["PreEOTMass"].append(km["PreEOTMass"])
                                        self.stats_per_block[block_name]["Entropy"].append(km["Entropy"])

                                if self.log_stats:
                                    if hasattr(self, "_batch_comp") and self._batch_comp:
                                        N_uncond = self._batch_comp.get("uncond", 0)
                                        N_pos    = self._batch_comp.get("pos", 0)
                                        idx_uncond_end = N_uncond * H
                                        idx_pos_end    = idx_uncond_end + N_pos * H
                                        
                                        if idx_uncond_end > 0:
                                            self._accumulate_from_probs_stream(attn_probs[0:idx_uncond_end], "uncond")
                                        if idx_pos_end > idx_uncond_end:
                                            self._accumulate_from_probs_stream(attn_probs[idx_uncond_end:idx_pos_end], "positive")
                                        if attn_probs.shape[0] > idx_pos_end:
                                            self._accumulate_from_probs_stream(attn_probs[idx_pos_end:], "negative")
                                    else:
                                        if attn_probs.shape[0] == 2 * H:

                                            self.set_stream_indices("uncond", self._global_batch_token_ids, self._global_pre_eot_index, self._global_eot_index)
                                            self._accumulate_from_probs_stream(attn_probs[0:H], "uncond")
                                            self.set_stream_indices("positive", self._global_batch_token_ids, self._global_pre_eot_index, self._global_eot_index)
                                            self._accumulate_from_probs_stream(attn_probs[H:], "positive")
                                        else:
                                            self.set_stream_indices("positive", self._global_batch_token_ids, self._global_pre_eot_index, self._global_eot_index)
                                            self._accumulate_from_probs_stream(attn_probs, "positive")

                        out = torch.bmm(attn_probs, v)
                        out = batch_to_head(out, bsz, q_len)
                        out = self_attn.to_out[0](out)
                        out = self_attn.to_out[1](out) if len(self_attn.to_out) > 1 else out
                        return out

                    return wrapped

                wrapped_fn = make_wrapped(module)
                module.forward = types.MethodType(wrapped_fn, module)

        self._attached = True

    def detach(self, unet):
        if not self._attached:
            return
        for name, module in unet.named_modules():
            if name in self._orig_forwards:
                module.forward = self._orig_forwards[name]
        self._orig_forwards.clear()
        self._attached = False

    def finalize(self) -> Dict[str, List[float]]:
        return self.stats


class LocalStableDiffusionPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,  
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        self.cos_sims = []
        self.cond_norms = []
        self.last_eot_info = None

    @torch.no_grad()
    def _compute_ref_eot_input_embedding(self, prompt: str, good_suffix: str):
        """
        Returns a single vector (H,) which is the *INPUT EMBEDDING* at EOT
        when tokenizing: prompt + good_suffix.
        """
        text = prompt + good_suffix
        print(f'------ ref prompt: {text}')
        enc = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.text_encoder.device) for k, v in enc.items()}
        input_ids = enc["input_ids"] 

        ids = input_ids[0].tolist()
        try:
            eot_idx = ids.index(self.tokenizer.eos_token_id)  
        except ValueError:
            pad_id = self.tokenizer.pad_token_id
            eot_idx = max(i for i, t in enumerate(ids) if t != pad_id)
        
        tok_emb = self.text_encoder.get_input_embeddings() 
        input_embs = tok_emb(input_ids) 
        
        ref_eot_input_vec = input_embs[0, eot_idx, :].detach().clone()  # (H,)
        return ref_eot_input_vec, eot_idx

    @torch.no_grad()
    def _compute_ref_eot_hidden(self, prompt: str, good_suffix: str):
        """
        Returns a single vector (H,) which is the *final hidden state* at EOT + padding
        when encoding: prompt + good_suffix.
        """
        text = prompt + good_suffix
        print(f'------ ref prompt: {text}')
        enc = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.text_encoder.device) for k, v in enc.items()}
        hidden = self.text_encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
        ).last_hidden_state 

        ids = enc["input_ids"][0].tolist()
        try:
            eot_idx = ids.index(self.tokenizer.eos_token_id)  
        except ValueError:
            pad_id = self.tokenizer.pad_token_id
            eot_idx = max(i for i, t in enumerate(ids) if t != pad_id)

        ref_eot_tail_slice = hidden[0, eot_idx:, :].detach().clone()  
        return ref_eot_tail_slice, eot_idx


    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        gt_prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        collect_step=1,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        callback=None,
        callback_steps=1,
        cross_attention_kwargs=None,
        track_noise_norm=False,
        lp=2,
        xattn_log: bool = False,
        xattn_save_json: Optional[str] = None,
        trigger_token_ids: Optional[List[int]] = None,  
        tail_token_ids: Optional[List[int]] = None,  
        gate_suffix: float = 1.0,  
        gate_tail: float = 1.0,    
        gate_window: str = "all", 
        gate_frac: float = 0.25,  
        log_loaded_mass: bool = False,  
        eot_shrink_alpha: float = 0.0,  
        eot_noise_sigma: float = 0.0, 
        eot_mean_pool: bool = False,  

        bias_schedule: str = "constant",    
        bias_start_frac: float = 0.0, 
        bias_end_frac: float = 1.0, 
        intervention_blocks: Optional[List[str]] = None, 

        spike_threshold: float = 0.0,  
        spike_penalty: float = 0.0,  
        spike_scale: float = 1.0,  
                
        pst_hot_map: Optional[Dict[str, List[int]]] = None,
        eot_hot_map: Optional[Dict[str, List[int]]] = None,
        bot_hot_map: Optional[Dict[str, List[int]]] = None,
        dynamic_head_selection: bool = False,
        dynamic_scout_steps: int = 5,
        dynamic_top_k_per_block: float = 3.0,
        dynamic_top_k_global: float = 5.0,
        dynamic_metric: str = "eot_mass_peakiness",
        dynamic_min_threshold: float = 0.0,
        trace_per_block: bool = True,
        log_stats: bool = True,

        causal_reencode_on_eot_noise: bool = True, 
        eot_blend_alpha: Optional[float] = None,
        
        ref_eot_vec: Optional[torch.Tensor] = None,
        ref_eot_input_vec: Optional[torch.Tensor] = None,
        disable_cfg_steps: int = 0,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        eot_idx = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if gt_prompt is not None:
            gt_prompt_embeds = self._encode_prompt(
                gt_prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
            
            if do_classifier_free_guidance:
                _, prompt_embeds_test = prompt_embeds.chunk(2)
                _, gt_prompt_embeds_test = gt_prompt_embeds.chunk(2)

            prompt_mean = prompt_embeds_test.mean(dim=1)   
            gt_mean     = gt_prompt_embeds_test.mean(dim=1) 

            cos_sim = torch.cosine_similarity(gt_mean, prompt_mean, dim=-1)
            self.cos_sims.append(cos_sim.detach().cpu().tolist())

        batch_token_ids = None
        toks = None
        try:
            if isinstance(prompt, (list, tuple)):
                toks = self.tokenizer(prompt[0], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            else:
                toks = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            batch_token_ids = toks.input_ids[0].tolist()  # [K]

            cond_input_ids = toks["input_ids"]           
            cond_attention_mask = toks["attention_mask"] 

        except Exception:
            batch_token_ids = None  
            toks = None

        xlog = None
        if xattn_log:
            if hasattr(self, "_xattn_logger") and self._xattn_logger: 
                self._xattn_logger._batch_comp = None
            if not hasattr(self, "_xattn_logger") or self._xattn_logger is None:
                self._xattn_logger = _XAttnLogger(
                    tokenizer=self.tokenizer,
                    trigger_token_ids=trigger_token_ids or [],
                    tail_token_ids=tail_token_ids or [],
                    gate_suffix=gate_suffix,
                    gate_tail=gate_tail,

                    gate_window=gate_window,
                    gate_frac=gate_frac,
                    enable_loaded_mass=log_loaded_mass,
                    
                    bias_schedule=bias_schedule,
                    bias_start_frac=bias_start_frac,
                    bias_end_frac=bias_end_frac,
                    intervention_blocks=intervention_blocks,
                    bias_on=bias_on,
                    spike_threshold=spike_threshold,
                    spike_penalty=spike_penalty,
                    spike_scale=spike_scale,
                    pst_hot_map=pst_hot_map,
                    eot_hot_map=eot_hot_map,
                    bot_hot_map=bot_hot_map,
                    dynamic_head_selection=dynamic_head_selection,
                    dynamic_scout_steps=dynamic_scout_steps,
                    dynamic_top_k_per_block=dynamic_top_k_per_block,
                    dynamic_top_k_global=dynamic_top_k_global,
                    dynamic_metric=dynamic_metric,
                    dynamic_min_threshold=dynamic_min_threshold,
                    trace_per_block=trace_per_block,
                    log_stats=log_stats,
                )
            xlog = self._xattn_logger
            xlog.set_batch_token_ids(batch_token_ids)

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)

            pre_eot_idx = None
            eot_idx = None
            old_pre_eot_idx = None   
            pst_idx = None   
            
            if batch_token_ids is not None:
                pre_eot_idx, eot_idx = find_pre_eot_and_eot(batch_token_ids, self.tokenizer)
                
                if gt_prompt is not None:
                    gt_toks = self.tokenizer(gt_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    gt_token_ids = gt_toks.input_ids[0].tolist()
                    
                    old_pre_eot_idx, pst_idx = find_pre_eot_and_eot(gt_token_ids, self.tokenizer)
                else:
                    pst_idx = pre_eot_idx

            xlog.set_batch_token_ids(batch_token_ids)
            xlog.set_special_indices(pre_eot=pst_idx, eot=eot_idx)
            xlog.attach(self.unet)

        if xlog is not None:
            xlog.start_run(label="positive")

        if eot_idx is not None and prompt_embeds is not None:
            self.last_eot_info = None
            
            if do_classifier_free_guidance:
                uncond_embeds, cond_embeds = prompt_embeds.chunk(2) 
            else:
                uncond_embeds, cond_embeds = None, prompt_embeds

            eot_before = cond_embeds[:, eot_idx, :].detach().clone()    
            tail_slice = slice(eot_idx, cond_embeds.shape[1])

            if eot_mean_pool:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                content_slice = slice(0, (pre_eot_idx + 1) if pre_eot_idx is not None else eot_idx)
                content = cond_embeds[:, content_slice, :]  # [B, L, H]
                if pad_id is not None and batch_token_ids is not None:
                    ids = torch.tensor(batch_token_ids, device=cond_embeds.device)
                    mask = (ids[content_slice] != pad_id).float()[None, :, None]
                    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    pooled = (content * mask).sum(dim=1, keepdim=True) / denom
                else:
                    pooled = content.mean(dim=1, keepdim=True)
                cond_embeds[:, eot_idx:eot_idx+1, :] = pooled  

            prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0) if do_classifier_free_guidance else cond_embeds

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if track_noise_norm is True:
            uncond_noise_norm = []
            text_noise_norm = []

            for i in range(len(latents)):
                uncond_noise_norm.append([])
                text_noise_norm.append([])

        # 7. Denoising loop
        base_prompt_embeds = prompt_embeds
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                current_do_cfg = do_classifier_free_guidance
                current_guidance_scale = guidance_scale
                
                if i < disable_cfg_steps:
                    current_do_cfg = False
                    current_guidance_scale = 0.0 

                if current_do_cfg:
                    latent_model_input = torch.cat([latents] * 2)
                    current_prompt_embeds = prompt_embeds
                else:
                    latent_model_input = latents
                    current_prompt_embeds = prompt_embeds.chunk(2)[0]

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if xlog is not None:
                    xlog.set_schedule(step_idx=i, num_steps=len(timesteps))
                
                embeds_for_step = base_prompt_embeds
                if batch_token_ids is not None:
                    if gate_window == "all" or (gate_window == "late" and i >= int((1.0 - gate_frac) * len(timesteps))):
                        gate_vec = torch.ones(len(batch_token_ids), device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                        for ti, tid in enumerate(batch_token_ids):
                            if tail_token_ids and tid in set(tail_token_ids):
                                gate_vec[ti] *= gate_tail
                            if trigger_token_ids and tid in set(trigger_token_ids):
                                gate_vec[ti] *= gate_suffix

                        if prompt_embeds.dim() == 3 and prompt_embeds.shape[1] == len(batch_token_ids):
                            embeds_for_step = base_prompt_embeds * gate_vec.unsqueeze(0).unsqueeze(-1)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if xlog is not None:
                    xlog._flush_step_stream("uncond")
                    xlog._flush_step_stream("positive")

                    if trace_per_block:
                        xlog._update_basin_state(step_idx=i)

                if current_do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) 
                    noise_pred_text = noise_pred_text - noise_pred_uncond
                    flat = noise_pred_text.view(noise_pred_text.shape[0], -1)
                    per_image_norms = flat.norm(p=2, dim=1)
                    mean_norm = per_image_norms.mean().item()
                    self.cond_norms.append(mean_norm)

                    noise_pred = noise_pred_uncond + current_guidance_scale * noise_pred_text

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if track_noise_norm is True:
                    for j in range(len(uncond_noise_norm)):
                        uncond_noise_norm[j].append(
                            noise_pred_uncond[j].norm(p=lp).item()
                        )
                        text_noise_norm[j].append(noise_pred_text[j].norm(p=lp).item())

        if xlog is not None:
            xlog.end_run(label="positive")
        
        if xlog is not None:
            try:
                self.last_xattn_stats = copy.deepcopy(getattr(xlog, "runs", []))
            except Exception as e:
                print(f"[xattn] finalize failed: {e}")
                self.last_xattn_stats = None

            try:
                xlog.detach(self.unet)
            except Exception as e:
                print(f"[xattn] detach failed: {e}")

            if xattn_save_json and self.last_xattn_stats:
                try:
                    with open(xattn_save_json, "w") as f:
                        json.dump(self.last_xattn_stats, f, indent=2)
                except Exception as e:
                    print(f"[xattn] Failed to save {xattn_save_json}: {e}")
        else:
            self.last_xattn_stats = None
        
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        if track_noise_norm is True:
            track_stats = {
                "uncond_noise_norm": uncond_noise_norm,
                "text_noise_norm": text_noise_norm,
            }
            return (
                StableDiffusionPipelineOutput(
                    images=image, nsfw_content_detected=has_nsfw_concept
                ),
                track_stats,
            )
        else:
            return StableDiffusionPipelineOutput(
                images=image, nsfw_content_detected=has_nsfw_concept
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def get_text_cond_grad(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        target_steps=[0],
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        all_token_grads = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if i in target_steps:
                    single_prompt_embeds = prompt_embeds[[0], :, :].clone().detach()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[-1], :, :].clone()

                    input_prompt_embeds = torch.cat(
                        [
                            dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                            single_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                        ]
                    )

                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=input_prompt_embeds,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_text = noise_pred_text - noise_pred_uncond
                    noise_pred_text_norm = torch.norm(noise_pred_text, p=2).mean()
                    loss = noise_pred_text_norm

                    (token_grads,) = torch.autograd.grad(loss, [prompt_embeds]) 
                    token_grads = token_grads.norm(p=2, dim=-1).mean(dim=0).detach()
                    all_token_grads.append(token_grads)

                    with torch.no_grad():
                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                    if i == max(target_steps):
                        torch.cuda.empty_cache()
                        return torch.mean(torch.stack(all_token_grads), dim=0)
                else:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond
                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )

                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                progress_bar.update()

    def aug_prompt(
        self,
        prompt=None,
        gt_prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        target_steps=[0],
        lr=0.1,
        optim_iters=10,
        target_loss=None,
        print_optim=False,
        optim_epsilon=None,
        clipping_threshold=None, 
        pruning_threshold=None, 
        xattn_log: bool = False,
        xattn_save_json: Optional[str] = None,
        trigger_token_ids: Optional[List[int]] = None,
        tail_token_ids: Optional[List[int]] = None,
        gate_suffix: float = 1.0,
        gate_tail: float = 1.0,
        gate_window: str = "all",
        gate_frac: float = 0.25,
        log_loaded_mass: bool = False,
        # pst_bias: float = 0.0,
        # eot_bias: float = 0.0,
        # bot_bias: float = 0.0,
        bias_schedule: str = "constant",
        bias_start_frac: float = 0.0,
        bias_end_frac: float = 1.0,
        intervention_blocks: Optional[List[str]] = None,
        bias_on: str = "cond",
        pst_hot_map: Optional[Dict[str, List[int]]] = None,
        eot_hot_map: Optional[Dict[str, List[int]]] = None,
        bot_hot_map: Optional[Dict[str, List[int]]] = None,
        dynamic_head_selection: bool = False,
        dynamic_scout_steps: int = 5,
        dynamic_top_k_per_block: float = 3.0,
        dynamic_top_k_global: float = 5.0,
        dynamic_metric: str = "eot_mass_peakiness",
        dynamic_min_threshold: float = 0.0,
        trace_per_block: bool = True,
        log_stats: bool = True,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        with torch.no_grad():
            # Prune
            if pruning_threshold is not None:
                small_mask = prompt_embeds.abs() < pruning_threshold
                pruned_count = small_mask.sum().item()
                total_count = prompt_embeds.numel()
                print(f"[Initial] Pruned weights: {pruned_count}/{total_count} ({pruned_count/total_count:.2%})")
                prompt_embeds[small_mask] = 0.0

            # Clip
            if clipping_threshold is not None:
                clipped_lower = (prompt_embeds <= -clipping_threshold).sum().item()
                clipped_upper = (prompt_embeds >= clipping_threshold).sum().item()
                total_clipped = clipped_lower + clipped_upper
                total_count = prompt_embeds.numel()
                print(f"[Initial] Clipped weights: {total_clipped}/{total_count} ({total_clipped/total_count:.2%})")
                prompt_embeds.clamp_(-clipping_threshold, clipping_threshold)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 1. Safely extract tokens (fixing the crash if prompt is None)
        batch_token_ids = None

        xlog = None
        if xattn_log:
            # Re-use or create logger
            if hasattr(self, "_xattn_logger") and self._xattn_logger: 
                self._xattn_logger._batch_comp = None
            
            if not hasattr(self, "_xattn_logger") or self._xattn_logger is None:
                self._xattn_logger = _XAttnLogger(
                    tokenizer=self.tokenizer,
                    trigger_token_ids=trigger_token_ids or [],
                    tail_token_ids=tail_token_ids or [],
                    gate_suffix=gate_suffix,
                    gate_tail=gate_tail,
                    gate_window=gate_window,
                    gate_frac=gate_frac,
                    enable_loaded_mass=log_loaded_mass,
                    # pst_bias=pst_bias,
                    # eot_bias=eot_bias,
                    # bot_bias=bot_bias,
                    bias_schedule=bias_schedule,
                    bias_start_frac=bias_start_frac,
                    bias_end_frac=bias_end_frac,
                    intervention_blocks=intervention_blocks,
                    bias_on=bias_on,
                    pst_hot_map=pst_hot_map,
                    eot_hot_map=eot_hot_map,
                    bot_hot_map=bot_hot_map,
                    dynamic_head_selection=dynamic_head_selection,
                    dynamic_scout_steps=dynamic_scout_steps,
                    dynamic_top_k_per_block=dynamic_top_k_per_block,
                    dynamic_top_k_global=dynamic_top_k_global,
                    dynamic_metric=dynamic_metric,
                    dynamic_min_threshold=dynamic_min_threshold,
                    trace_per_block=trace_per_block,
                    log_stats=log_stats,
                )
            
            xlog = self._xattn_logger
            xlog.set_batch_token_ids(batch_token_ids)
            
            pre_eot_idx = None
            eot_idx = None
            pst_idx = None 

            if batch_token_ids is not None:
                pre_eot_idx, eot_idx = find_pre_eot_and_eot(batch_token_ids, self.tokenizer)
                if gt_prompt is not None:
                     try:
                        gt_toks = self.tokenizer(gt_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
                        gt_ids = gt_toks.input_ids[0].tolist()
                        _, pst_idx = find_pre_eot_and_eot(gt_ids, self.tokenizer)
                     except:
                        pst_idx = pre_eot_idx
                else:
                    pst_idx = pre_eot_idx
            
            xlog.set_special_indices(pre_eot=pst_idx, eot=eot_idx)
            xlog.attach(self.unet)
            xlog.start_run(label="aug_prompt_run")

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if xlog is not None:
                    xlog.set_schedule(step_idx=i, num_steps=len(timesteps))

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                skip_optimization = (target_steps is None or len(target_steps) == 0 or (target_steps == [-1]))
                if skip_optimization:
                    return prompt_embeds[[-1], :, :]

                if not skip_optimization and (i in target_steps):
                    single_prompt_embeds = prompt_embeds[[-1], :, :].clone().detach()
                    if print_optim is True or optim_epsilon is not None:
                        init_embeds = single_prompt_embeds.clone()
                    single_prompt_embeds.requires_grad = True
                    dummy_prompt_embeds = prompt_embeds[[0], :, :].clone()   # <-- Original code     
                    # optimizer
                    optimizer = torch.optim.AdamW([single_prompt_embeds], lr=lr)

                    prompt_tokens = self.tokenizer.encode(prompt)
                    prompt_tokens = prompt_tokens[1:-1]
                    prompt_tokens = prompt_tokens[:75]

                    curr_learnabel_mask = list(set(range(77)) - set([0]))

                    for j in range(optim_iters):
                        if print_optim is True or optim_epsilon is not None:
                            with torch.no_grad():
                                tmp_init_embeds = init_embeds[:, curr_learnabel_mask]
                                tmp_init_embeds = tmp_init_embeds.reshape(
                                    -1, tmp_init_embeds.shape[-1]
                                )
                                tmp_single_prompt_embeds = single_prompt_embeds[
                                    :, curr_learnabel_mask
                                ]
                                tmp_single_prompt_embeds = (
                                    tmp_single_prompt_embeds.reshape(
                                        -1, tmp_single_prompt_embeds.shape[-1]
                                    )
                                )

                                l_inf = torch.norm(
                                    tmp_init_embeds - tmp_single_prompt_embeds,
                                    p=float("inf"),
                                    dim=-1,
                                ).mean()
                                l_2 = torch.norm(
                                    tmp_init_embeds - tmp_single_prompt_embeds,
                                    p=2,
                                    dim=-1,
                                ).mean()

                        input_prompt_embeds = torch.cat(
                            [
                                dummy_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                                single_prompt_embeds.repeat(
                                    num_images_per_prompt, 1, 1
                                ),
                            ]
                        )

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=input_prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        # separate uncond vs. text
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond
                        noise_pred_text_norm = torch.norm(noise_pred_text, p=2).mean()

                        loss = noise_pred_text_norm
                        loss_item = loss.item()
                          
                        if target_loss is not None:
                            if loss_item <= target_loss:
                                if print_optim is True:
                                    print(f"Early stop at step: {j}, curr loss: {loss_item}")
                                break

                        (single_prompt_embeds.grad,) = torch.autograd.grad(
                            loss, [single_prompt_embeds]
                        )
                        single_prompt_embeds.grad[:, [0]] = (
                            single_prompt_embeds.grad[:, [0]] * 0
                        )

                        optimizer.step()
                        optimizer.zero_grad()

                        if print_optim is True:
                            print(f"step: {j}, curr loss: {loss_item}")

                    single_prompt_embeds = single_prompt_embeds.detach()
                    single_prompt_embeds.requires_grad = False
                    torch.cuda.empty_cache()

                    if xlog is not None:
                        xlog.end_run(label="aug_prompt_final")
                        
                        try:
                            self.last_xattn_stats = copy.deepcopy(getattr(xlog, "runs", []))
                        except: pass
                        xlog.detach(self.unet)
                        
                        if xattn_save_json and self.last_xattn_stats:
                            try:
                                with open(xattn_save_json, "w") as f:
                                    json.dump(self.last_xattn_stats, f, indent=2)
                            except: pass
                    
                    return single_prompt_embeds

                else:
                    print('oops run into this ')
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            return_dict=False,
                        )[0]

                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_text = noise_pred_text - noise_pred_uncond

                        noise_pred = (
                            noise_pred_uncond + guidance_scale * noise_pred_text
                        )

                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                progress_bar.update()

    @torch.no_grad()
    def dual_guidance_call(
        self,
        prompt_positive,
        prompt_negative_repulsive,
        guidance_scale_positive=7.5,
        guidance_scale_negative=3.0,
        negative_guidance_decay_schedule="sine", 
        positive_guidance_schedule="cosine",    

        neighbor_gating_mode="soft",
        neighbor_gating_threshold=0.9,
        neighbor_gating_soft_sigma=0.5,
        use_only_nearest_for_positive=False, 
        soft_per_neighbor=True,              

        orthogonalize_neighbors=True,
        ortho_cos_threshold=0.9,    
        ortho_coef_clip=2.0,  
        ortho_eps=1e-8,
        # td_reg: bool = True,
        # td_beta: float = 0.95,
        # td_eps: float = 1e-8,
        # td_tiny: float = 1e-6,

        height=None,
        width=None,
        num_inference_steps=50,
        num_images_per_prompt=1,
        eta=0.0,
        generator=None,
        latents=None,
        output_type="pil",
        return_dict=True,

        xattn_log: bool = False,
        xattn_save_json: Optional[str] = None,
        trigger_token_ids: Optional[List[int]] = None,
        tail_token_ids: Optional[List[int]] = None,
        gate_suffix: float = 1.0,
        gate_tail: float = 1.0,
        gate_window: str = "all",
        gate_frac: float = 0.25,
        log_loaded_mass: bool = False,
        eot_logit_bias_beta: float = 0.0,
        xattn_log_stream: str = "both",  
        trace_per_block: bool = True,
        log_stats: bool = True,
        basin_beta: float = 0.0,

        intervention_blocks: Optional[List[str]] = None,
        bias_on: str = "cond",
        spike_threshold: float = 0.0,
        spike_penalty: float = 0.0,
        spike_scale: float = 1.0,
        # pst_bias: float = 0.0,
        # eot_bias: float = 0.0,
        # bot_bias: float = 0.0,

    ):
        """
        Generates images using dual guidance.

        Args:
            prompt_positive (str): The prompt to guide towards (e.g., a neighbor prompt).
            prompt_negative_repulsive (str): The prompt to guide away from (e.g., the memorized prompt).
            guidance_scale_positive (float): Weight for the attractive force (w_pos).
            guidance_scale_negative (float): Weight for the repulsive force (w_neg).
            ... (other standard generation args)
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        image_batch_size = num_images_per_prompt
        device = self._execution_device


        text_inputs_uncond = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings_uncond = self.text_encoder(text_inputs_uncond.input_ids.to(self.device))[0]
        uncond_embeds = text_embeddings_uncond.repeat(image_batch_size, 1, 1)

        if isinstance(prompt_positive, (list, tuple)):
            neighbor_list = list(prompt_positive)
        else:
            neighbor_list = [prompt_positive]
        k_neighbors = len(neighbor_list)

        text_inputs_positive = self.tokenizer(neighbor_list, padding="max_length",
            max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings_positive = self.text_encoder(text_inputs_positive.input_ids.to(self.device))[0]   # [k, H]
        prompt_embeds_positive = text_embeddings_positive.repeat_interleave(num_images_per_prompt, dim=0) # [k*N, H]

        text_inputs_negative_repulsive = self.tokenizer(
            [prompt_negative_repulsive], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_embeddings_negative_repulsive = self.text_encoder(text_inputs_negative_repulsive.input_ids.to(self.device))[0]
        prompt_embeds_negative = text_embeddings_negative_repulsive.repeat(image_batch_size, 1, 1)

        def masked_mean_pool(x, input_ids):
            pad_id = self.tokenizer.pad_token_id
            mask = (input_ids != pad_id).float().to(x.device)
            mask = mask.unsqueeze(-1)                         
            x_sum = (x * mask).sum(dim=1)                     
            denom = mask.sum(dim=1).clamp_min(1e-8)         
            return x_sum / denom   

        pooled_pos = masked_mean_pool(text_embeddings_positive.float(),
                                    text_inputs_positive.input_ids)
        pooled_gt  = masked_mean_pool(text_embeddings_negative_repulsive.float(),
                                    text_inputs_negative_repulsive.input_ids)

        pooled_pos_n = F.normalize(pooled_pos, dim=-1)
        pooled_gt_n  = F.normalize(pooled_gt,  dim=-1)
        cos_sims     = torch.matmul(pooled_pos_n, pooled_gt_n.t()).squeeze(-1).clamp(-1.0, 1.0)  # [k]
        sim_max, idx_max = torch.max(cos_sims, dim=0)

        if neighbor_gating_mode == "hard":
            tau = float(neighbor_gating_threshold)
            if use_only_nearest_for_positive:
                if sim_max >= tau:
                    keep_idx = idx_max.view(1)
                else:
                    keep_idx = torch.empty(0, dtype=torch.long, device=cos_sims.device)
            else:
                keep_idx = torch.nonzero(cos_sims >= tau, as_tuple=False).squeeze(-1)
            print(f'keep indices: {keep_idx.cpu().numpy().tolist()}, threshold: {tau}')

            if keep_idx.numel() == 0:
                # Deactivate positive branch
                k_neighbors = 0
                prompt_embeds_positive = text_embeddings_uncond.repeat(0, 1, 1)
                g_pos = 0.0
            else:
                text_embeddings_positive = text_embeddings_positive.index_select(0, keep_idx)
                prompt_embeds_positive   = text_embeddings_positive.repeat_interleave(num_images_per_prompt, dim=0)
                k_neighbors = text_embeddings_positive.shape[0]
                g_pos = 1.0

        elif neighbor_gating_mode == "soft":
            if use_only_nearest_for_positive:
                text_embeddings_positive = text_embeddings_positive[idx_max:idx_max+1]
                prompt_embeds_positive   = text_embeddings_positive.repeat_interleave(num_images_per_prompt, dim=0)
                k_neighbors = 1
                tau   = float(neighbor_gating_threshold)
                sigma = max(1e-6, float(neighbor_gating_soft_sigma))
                deficit = torch.clamp(tau - sim_max, min=0.0)
                g_pos = torch.exp(- (deficit / sigma) ** 2).item()
                neighbor_weights = None  
            else:
                k_neighbors = len(neighbor_list)
                tau   = float(neighbor_gating_threshold)
                sigma = max(1e-6, float(neighbor_gating_soft_sigma))

                if soft_per_neighbor:
                    deficit_vec = torch.clamp(tau - cos_sims, min=0.0)         
                    neighbor_weights = torch.exp(- (deficit_vec / sigma) ** 2)  
                    g_pos = 1.0
                else:
                    neighbor_weights = None
                    deficit = torch.clamp(tau - sim_max, min=0.0)
                    g_pos = torch.exp(- (deficit / sigma) ** 2).item()


        else: 
            if use_only_nearest_for_positive:
                text_embeddings_positive = text_embeddings_positive[idx_max:idx_max+1]
                prompt_embeds_positive   = text_embeddings_positive.repeat_interleave(num_images_per_prompt, dim=0)
                k_neighbors = 1
            else:
                k_neighbors = len(neighbor_list)
            g_pos = 1.0

        print(f"[dual_guidance] selected neighbours --- [{neighbor_gating_mode}]:")
        if neighbor_gating_mode == "hard":
            sel = keep_idx.tolist() if keep_idx.numel() > 0 else []
            for i in sel:
                print(f"  similarity={float(cos_sims[i].item()):.4f} : {neighbor_list[i]!r}")
            
        elif neighbor_gating_mode == "soft" and not use_only_nearest_for_positive and soft_per_neighbor and k_neighbors > 0:
            for i in range(len(neighbor_list)):
                print(f"  similarity={float(cos_sims[i].item()):.4f} : {neighbor_list[i]!r} (weight={float(neighbor_weights[i].item()):.4f})")
        else:
            if use_only_nearest_for_positive:
                i = int(idx_max.item())
                print("[dual_guidance][soft] nearest:",
                    (neighbor_list[i], float(cos_sims[i])))
            else:
                print(f"  (no weights assigned based on similarity)")

        xlog = None
        if xattn_log or (eot_logit_bias_beta and eot_logit_bias_beta > 0.0):
            if not hasattr(self, "_xattn_logger") or self._xattn_logger is None:
                self._xattn_logger = _XAttnLogger(
                    tokenizer=self.tokenizer,
                    trigger_token_ids=trigger_token_ids or [],
                    tail_token_ids=tail_token_ids or [],
                    gate_suffix=gate_suffix,
                    gate_tail=gate_tail,
                    gate_window=gate_window,
                    gate_frac=gate_frac,
                    enable_loaded_mass=log_loaded_mass,
                    trace_per_block=trace_per_block,
                    log_stats=log_stats,
                    basin_beta=basin_beta,
                    # pst_bias=pst_bias,
                    # eot_bias=eot_bias,
                    # bot_bias=bot_bias,
                    intervention_blocks=intervention_blocks,
                    spike_threshold=spike_threshold, 
                    spike_penalty=spike_penalty,  
                    spike_scale=spike_scale,    
                )
            xlog = self._xattn_logger
            xlog.attach(self.unet) 

            toks_uncond = text_inputs_uncond.input_ids[0].detach().cpu().tolist()
            pre_eot_uncond, eot_uncond = find_pre_eot_and_eot(toks_uncond, self.tokenizer)
            xlog.set_stream_indices(
                label="uncond", 
                token_ids=toks_uncond, 
                pre_eot=pre_eot_uncond, 
                eot=eot_uncond
            )

            if k_neighbors > 0:
                toks_pos = text_inputs_positive.input_ids[0].detach().cpu().tolist()
                pre_eot_pos, eot_pos = find_pre_eot_and_eot(toks_pos, self.tokenizer)
                xlog.set_stream_indices(
                    label="positive",
                    token_ids=toks_pos,
                    pre_eot=pre_eot_pos,
                    eot=eot_pos
                )

            toks_neg = text_inputs_negative_repulsive.input_ids[0].detach().cpu().tolist()
            pre_eot_neg, eot_neg = find_pre_eot_and_eot(toks_neg, self.tokenizer)
            xlog.set_stream_indices(
                label="negative",
                token_ids=toks_neg,
                pre_eot=pre_eot_neg,
                eot=eot_neg
            )

            if xattn_log_stream == "positive" and k_neighbors > 0:
                xlog.set_batch_token_ids(toks_pos) 
                xlog.set_special_indices(pre_eot=pre_eot_pos, eot=eot_pos) 
            else:
                xlog.set_batch_token_ids(toks_neg) 
                xlog.set_special_indices(pre_eot=pre_eot_neg, eot=eot_neg)

            xlog.start_run(label="uncond")
            xlog.start_run(label="positive")
            xlog.start_run(label="negative")


        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            image_batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds_positive.dtype,
            device,
            generator,
            latents,
        )
        N = latents.shape[0] 

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if xlog is not None:
                    xlog.set_schedule(step_idx=i, num_steps=len(timesteps))
                    xlog.set_batch_composition(N_uncond=N, N_pos=k_neighbors * N, N_neg=N)      
                    xlog._ensure_label_accum("uncond")
                    xlog._ensure_label_accum("positive")
                    xlog._ensure_label_accum("negative")

                latent_model_input = torch.cat([latents] * (2 + k_neighbors), dim=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                if k_neighbors == 0:
                    empty_pos = text_embeddings_uncond.repeat(0, 1, 1)
                    prompt_embeds_batch = torch.cat([uncond_embeds, empty_pos, prompt_embeds_negative], dim=0)
                else:
                    prompt_embeds_batch = torch.cat([
                        uncond_embeds,                    # [N, H]
                        prompt_embeds_positive,          # [k*N, H]
                        prompt_embeds_negative  # [N, H]
                    ], dim=0)
        

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_batch,
                    return_dict=False,
                )[0]

                if xlog is not None and log_stats:
                    xlog._flush_step_stream("uncond")
                    xlog._flush_step_stream("positive")
                    xlog._flush_step_stream("negative")

                noise_pred_uncond, rest = torch.split(noise_pred, [N, k_neighbors * N + N], dim=0)
                noise_pred_neighbors_flat, noise_pred_negative = torch.split(rest, [k_neighbors * N, N], dim=0)

                C, Hh, Ww = noise_pred_uncond.shape[1:]
                v_u  = noise_pred_uncond                         
                v_gt = noise_pred_negative - noise_pred_uncond    
                v_nb = torch.zeros_like(noise_pred_uncond)     

                if k_neighbors > 0:
                    noise_pred_positive = noise_pred_neighbors_flat.view(k_neighbors, N, C, Hh, Ww)
                    v_nbs = noise_pred_positive - noise_pred_uncond.unsqueeze(0)  # [k, N, C, H, W]
                    if ('neighbor_weights' in locals()) and (neighbor_weights is not None):
                        w = neighbor_weights.view(k_neighbors, 1, 1, 1, 1).to(v_nbs.dtype)
                        denom = w.sum().clamp_min(1e-8)
                        v_nb = (w * v_nbs).sum(dim=0) / denom     
                    else:
                        v_nb = v_nbs.mean(dim=0)  

                if orthogonalize_neighbors:
                    vn32 = v_nb.float().reshape(N, -1)  
                    vg32 = v_gt.float().reshape(N, -1)  

                    dot = (vn32 * vg32).sum(dim=1, keepdim=True)    
                    vg_norm2 = (vg32 * vg32).sum(dim=1, keepdim=True) + ortho_eps  
                    coef = dot / vg_norm2   

                    vn_norm = torch.sqrt((vn32 * vn32).sum(dim=1, keepdim=True) + ortho_eps)
                    vg_norm = torch.sqrt((vg32 * vg32).sum(dim=1, keepdim=True) + ortho_eps)
                    cos = (dot / (vn_norm * vg_norm)).clamp(-1.0, 1.0)  
                    mask = (cos > ortho_cos_threshold).float()    

                    coef = coef.clamp(min=-ortho_coef_clip, max=ortho_coef_clip)

                    coef_b = coef.view(N, 1, 1, 1)
                    mask_b = mask.view(N, 1, 1, 1)

                    v_n_perp = v_nb - mask_b * (coef_b * v_gt)   
                    v_n_eff = v_n_perp.to(v_nb.dtype)
                else:
                    v_n_eff = v_nb

                progress = i / len(timesteps)

                if xlog is not None and trace_per_block:
                    xlog._update_basin_state(step_idx=i)

                is_in_basin = True
                if xlog is not None:
                    is_in_basin = xlog.in_basin

                if negative_guidance_decay_schedule == "linear":
                    current_w_neg = guidance_scale_negative * (1 - progress)
                elif negative_guidance_decay_schedule == "cosine":
                    current_w_neg = guidance_scale_negative * math.cos(progress * math.pi / 2)
                elif negative_guidance_decay_schedule == "sine":
                    current_w_neg = guidance_scale_negative * math.sin(progress * math.pi / 2)                
                else:  
                    current_w_neg = guidance_scale_negative
                
                if not is_in_basin:
                    current_w_neg = 0.0
                    
                if positive_guidance_schedule == "none":
                    time_fade = 1.0
                elif positive_guidance_schedule =="linear":
                    time_fade = 1.0 - progress
                elif positive_guidance_schedule == "cosine":
                    time_fade = math.cos(progress * math.pi / 2.0)
                elif positive_guidance_schedule =="cos2":
                    time_fade = math.cos(progress * math.pi / 2.0) ** 2
                else:
                    time_fade = 1.0

                current_w_pos = guidance_scale_positive * g_pos * time_fade


                noise_pred = v_u + current_w_pos * v_n_eff - current_w_neg * v_gt 
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                progress_bar.update()

        if xlog is not None:
            try:
                xlog.end_run(label="uncond")
                xlog.end_run(label="positive")
                xlog.end_run(label="negative")
                
                self.last_xattn_stats_dual = copy.deepcopy(getattr(xlog, "runs", []))
            except Exception as e:
                print(f"[xattn-dual] finalize failed: {e}")
                self.last_xattn_stats_dual = None

            try:
                xlog.detach(self.unet)
            except Exception as e:
                print(f"[xattn-dual] detach failed: {e}")

            if xattn_save_json and self.last_xattn_stats_dual:
                try:
                    with open(xattn_save_json, "w") as f:
                        json.dump(self.last_xattn_stats_dual, f, indent=2)
                except Exception as e:
                    print(f"[xattn-dual] Failed to save {xattn_save_json}: {e}")

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds_positive.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )



    def _compute_clip_score(self, image, prompt, clip_model, clip_processor):
        """
        Compute a CLIP score between an image (decoded from latents) and a text prompt.
        This function assumes `image` is in the expected format for the clip_processor.
        """
        inputs = clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self._execution_device)  

        outputs = clip_model(**inputs)
        return outputs.logits_per_image.mean()

    def _compute_clip_score(self, image, prompt, clip_model, clip_preprocess, tokenizer):
        """
        Compute a CLIP similarity score between an image and a text prompt.
        This function mimics the working measure_CLIP_similarity, using a preprocess function and tokenizer.
        """
        import numpy as np
        from PIL import Image

        if isinstance(image, torch.Tensor):
            if image.ndim == 4:  
                image = image[0]  
            image = image.to(torch.float32)  
            image_np = image.detach().cpu().numpy()
            
            image_np = np.squeeze(image_np)
            
            if image_np.ndim == 2:
                pass
            elif image_np.ndim == 3:
                if image_np.shape[0] in [1, 3]:
                    image_np = image_np.transpose(1, 2, 0)  
            else:
                raise ValueError("Unexpected image shape after squeezing: " + str(image_np.shape))
            
            image = Image.fromarray(image_np)
        
        img_tensor = clip_preprocess(image).unsqueeze(0).to(self._execution_device)
        text_tokens = tokenizer([prompt]).to(self._execution_device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(img_tensor)
            text_features = clip_model.encode_text(text_tokens)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).mean(-1)
        return similarity

    def _compute_lpips_score(self, img_tensor, ref_tensor, lpips_model):
        return lpips_model(img_tensor, ref_tensor).mean().item()

