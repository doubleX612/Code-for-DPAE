import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

def RMS(sig):
    samples,lens = sig.shape
    rms_res = np.zeros(samples)
    for i in range(samples):
        rms_res[i] = math.sqrt(sum([x ** 2 for x in sig[i,:]]) / len(sig[i,:]))
    return rms_res

def RMS_temporal(pred,ground):
    rms_numerator = RMS(pred-ground)
    rms_denominator = RMS(ground)
    return rms_numerator/rms_denominator

def RMS_spectral(pred,ground):
    psd_pred,psd_ground =[],[]
    for i in range(pred.shape[0]):
        f, psd_spec = signal.welch(pred[i], 512, 'flattop', 512, scaling='spectrum')
      
        psd_pred.append(psd_spec.reshape(-1))
        f, psd_spec = signal.welch(ground[i], 512, 'flattop', 512, scaling='spectrum')
        psd_ground.append(psd_spec.reshape(-1))
    psd_pred,psd_ground = np.array(psd_pred),np.array(psd_ground)
    rms_numerator = RMS(psd_pred-psd_ground)
    rms_denominator = RMS(psd_ground)
    return rms_numerator/rms_denominator
    
def CC(pred,ground):
    cc_res = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
#         cov_mat = np.corrcoef(pred[i,:], ground[i,:])
#         cc_res[i] = cov_mat[0,1]
        cc_res[i],temp = pearsonr(pred[i,:],ground[i,:])
    return cc_res

def plot_data(denoised,clean,clip_size):
#     clip_size = 560
    tem_value = np.zeros(10)
    spec_value = np.zeros(10)
    cc_value = np.zeros(10)
    for i in range(10):
        test_pre,test_ground = denoised[clip_size*i:clip_size*(i+1),:],clean[clip_size*i:clip_size*(i+1),:]
        tem_value[i] = np.mean(RMS_temporal(test_pre,test_ground))
        spec_value[i] = np.mean(RMS_spectral(test_pre,test_ground))
        cc_value[i] = np.mean(CC(test_pre,test_ground))
        print(i)

    return tem_value,spec_value,cc_value
