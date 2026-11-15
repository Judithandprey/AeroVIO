"""Frozen motion-regime definitions; all truth use is evaluation-only."""
import numpy as np
from viobench.tools.geometry import poses_to_matrices

def cruise_metrics(truth,pred,times,config):
    n=config['window_frames'];dt=np.diff(times)
    assert np.allclose(dt,.05,atol=1e-6)
    delta=np.diff(truth[:,:3],axis=0);speed=np.linalg.norm(delta,axis=1)/dt
    rotation=poses_to_matrices(truth[:,:7])[:,:3,:3]
    d=rotation[:-1].transpose(0,2,1)@rotation[1:]
    angular=np.degrees(np.arccos(np.clip((np.trace(d,axis1=1,axis2=2)-1)/2,-1,1)))/dt
    speeds=np.lib.stride_tricks.sliding_window_view(speed,n)
    turns=np.lib.stride_tricks.sliding_window_view(angular,n)
    actual=np.linalg.norm(truth[n:,:3]-truth[:-n,:3],axis=1)
    estimate=np.linalg.norm(pred[n:,:3]-pred[:-n,:3],axis=1)
    path=np.lib.stride_tricks.sliding_window_view(np.linalg.norm(delta,axis=1),n).sum(1)
    mean=speeds.mean(1)
    mask=(mean>=config['cruise_min_mean_speed'])&(speeds.std(1)/np.maximum(mean,1e-12)<=config['cruise_max_speed_cv'])
    mask&=(np.ptp(speeds,axis=1)<=config['cruise_max_speed_range'])&(actual/np.maximum(path,1e-12)>=config['cruise_min_path_efficiency'])
    mask&=turns.mean(1)<config['cruise_max_mean_turn_deg_s']
    errors=estimate[mask]-actual[mask]
    predicted_speed=np.linalg.norm(np.diff(pred[:,:3],axis=0),axis=1)/dt
    false=(speed>=config['cruise_min_mean_speed'])&(predicted_speed<config['stationary_speed'])
    events=[];i=0
    while i<len(false):
        if not false[i]:i+=1;continue
        end=i
        while end<len(false) and false[end]:end+=1
        if end-i>=config['false_stop_run_frames']:events.append({'start_frame':i,'duration_s':float(dt[i:end].sum())})
        i=end
    return {'windows':int(mask.sum()),'length_squared_error_sum':float((errors**2).sum()),
        'length_error_sum':float(errors.sum()),'length_rmse_m':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        'length_bias_m':float(np.mean(errors)) if len(errors) else None,'false_stop_events':events,
        'false_stop_test_scope':'all true-moving intervals >=0.1 m/s, not only cruise windows'}

def combine_cruise(records):
    n=sum(r['windows'] for r in records)
    return {'windows':n,'length_rmse_m':float(np.sqrt(sum(r['length_squared_error_sum'] for r in records)/n)) if n else None,
        'length_bias_m':sum(r['length_error_sum'] for r in records)/n if n else None,
        'false_stop_events':sum(len(r['false_stop_events']) for r in records),
        'false_stop_total_seconds':sum(e['duration_s'] for r in records for e in r['false_stop_events'])}
