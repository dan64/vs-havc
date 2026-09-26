import math
import numpy as np
import torch
from typing import Optional


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See appendix for derivation
        # or you can just trust me ヽ(ー_ー )ノ
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq+two_ab-b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq+two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*HW

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False,
               group_mask=None, group_distance=None, alpha_norm=None):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    # group_mask/group_distance/alpha_norm (proximity bias): same group_mask
    # contract: (B, N, HW) boolean, or an .expand()'ed view, gathered at the
    # selected top-k indices below, verified to work with torch.gather;
    # group_distance is the RAW temporal distance for the group's elements
    # (same shape contract as group_mask), zero/irrelevant elsewhere - the
    # penalty itself is not precomputed by the caller: it can only be
    # computed here, after top-k selection, because it is SCALED by the
    # spread (max-min) of the survivors' own raw scores - a quantity that
    # does not exist before this point. alpha_norm is a plain scalar (not a
    # tensor): the normalized penalty strength.
    # This does NOT change how much the group weighs in the final
    # distribution overall - it only redistributes *within* that fixed
    # aggregate weight, favoring less-penalized members. Default None ->
    # identical behavior to before this mechanism existed, for every other
    # caller (training included, which never passes it).
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        if group_mask is not None:
            is_group = torch.gather(group_mask, dim=1, index=indices)  # bool, (B, top_k, HW)

            # reference weights with NO bias at all - needed to know how
            # much the group is "supposed" to weigh in aggregate
            w_full = torch.softmax(values, dim=1)
            W_group = (w_full * is_group).sum(dim=1, keepdim=True)  # (B, 1, HW), preserved as-is

            # penalty = alpha_norm * distance_norm * range_group,
            # computed here (not by the caller) because both distance_norm
            # and range_group are relative to what actually survived top-k
            # at this exact query position, not the whole perm_mem window.
            dist = (torch.gather(group_distance, dim=1, index=indices)
                    if group_distance is not None else torch.zeros_like(values))
            has_group = is_group.any(dim=1, keepdim=True)  # (B, 1, HW)

            # range_group: max-min of the GROUP's own raw scores only.
            # Masked via explicit sentinels + an explicit has_group guard
            # (not relying on IEEE inf-inf arithmetic happening to behave -
            # verified empirically that it does NOT produce a clean, safe
            # value on its own for an empty group).
            vals_for_max = torch.where(is_group, values, torch.full_like(values, float('-inf')))
            vals_for_min = torch.where(is_group, values, torch.full_like(values, float('inf')))
            group_max = vals_for_max.max(dim=1, keepdim=True).values
            group_min = vals_for_min.min(dim=1, keepdim=True).values
            range_group = torch.where(has_group, group_max - group_min, torch.zeros_like(group_max))

            # distance_norm: distance / max(distance) among the group's own
            # survivors at this query position (dist_max_in_window) -
            # explicit fallback to 0 when dist_max_in_window==0 (empty
            # group, or every surviving group member at distance 0),
            # instead of dividing (required explicitly).
            dist_masked_for_max = torch.where(is_group, dist, torch.zeros_like(dist))
            dist_max_in_window = dist_masked_for_max.max(dim=1, keepdim=True).values  # (B, 1, HW)
            safe_denom = torch.clamp(dist_max_in_window, min=1e-12)
            distance_norm = torch.where(dist_max_in_window > 0, dist / safe_denom, torch.zeros_like(dist))

            penalty = (alpha_norm if alpha_norm is not None else 0.0) * distance_norm * range_group

            # softmax restricted to the group only, to decide the internal
            # split of that fixed W_group budget. Edge case (verified, not
            # just in theory): if is_group is all-False for a given HW
            # column, every entry is -inf, softmax's internal
            # max-subtraction gives -inf-(-inf)=NaN for the whole column.
            # W_group is exactly 0 there too, so the NaN would be multiplied
            # by 0 - not automatically 0 in IEEE float (0*NaN=NaN) - hence
            # the explicit nan_to_num before it ever reaches w_final.
            values_group_masked = torch.where(is_group, values - penalty, torch.full_like(values, float('-inf')))
            w_group_internal = torch.softmax(values_group_masked, dim=1)
            w_group_internal = torch.nan_to_num(w_group_internal, nan=0.0)

            x_exp = torch.where(is_group, w_group_internal * W_group, w_full)
        else:
            x_exp = values.exp_()
            x_exp /= torch.sum(x_exp, dim=1, keepdim=True)

        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, ms, qk, qe):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T*H*W) 
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem
