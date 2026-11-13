"""Training-only joint speed/turn balancing with explicit marginal and reuse caps."""
import numpy as np


def cell_quotas(counts, speed_quotas, max_repeats):
    counts=np.asarray(counts,dtype=np.int64)
    speed_quotas=np.asarray(speed_quotas,dtype=np.int64)
    if counts.ndim!=2 or speed_quotas.shape!=(len(counts),) or np.any(counts<0):
        raise ValueError('invalid support table')
    if max_repeats<1 or np.any(speed_quotas<0):
        raise ValueError('nonnegative quotas and positive reuse cap required')
    capacity=counts*max_repeats
    if np.any(capacity.sum(axis=1)<speed_quotas):
        raise ValueError('insufficient unique support for requested marginal under reuse cap')
    quotas=np.zeros_like(counts)
    for speed,quota in enumerate(speed_quotas):
        for _ in range(int(quota)):
            candidates=np.flatnonzero(quotas[speed]<capacity[speed])
            chosen=candidates[np.argmin(quotas[speed,candidates])]
            quotas[speed,chosen]+=1
    return quotas


def make_order(speed_ids,turn_ids,quotas,epoch,seed):
    speed_ids,turn_ids=np.asarray(speed_ids),np.asarray(turn_ids)
    rng=np.random.default_rng(seed+1000+epoch)
    parts=[]
    for s in range(quotas.shape[0]):
        for t in range(quotas.shape[1]):
            quota=int(quotas[s,t])
            if not quota:continue
            pool=np.flatnonzero((speed_ids==s)&(turn_ids==t))
            if not len(pool):raise ValueError('positive quota for empty cell')
            chunks=[rng.permutation(pool) for _ in range((quota+len(pool)-1)//len(pool))]
            parts.append(np.concatenate(chunks)[:quota])
    return rng.permutation(np.concatenate(parts)).astype(np.int64)


def audit_order(order,speed_ids,turn_ids,quotas,max_repeats):
    order=np.asarray(order);n=len(speed_ids)
    if order.ndim!=1 or order.dtype.kind not in 'iu' or len(order)!=n or np.any(order<0) or np.any(order>=n):
        raise ValueError('epoch must contain exactly one original epoch worth of valid indices')
    counts=np.zeros_like(quotas)
    np.add.at(counts,(speed_ids[order],turn_ids[order]),1)
    repeats=np.bincount(order,minlength=n)
    if not np.array_equal(counts,quotas) or repeats.max()>max_repeats:
        raise ValueError('exposure gate failed')
    return int(repeats.max())
