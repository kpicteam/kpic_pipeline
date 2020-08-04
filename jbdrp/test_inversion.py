import numpy as np


from scipy.optimize import lsq_linear
N = 100
sig = 1
sigmas_vec = np.ones(N)*sig
logdet_Sigma = np.sum(2 * np.log(sigmas_vec))
ravelHPFdata = np.random.randn(N)*sig+np.ones(N)
print(np.nanstd(ravelHPFdata),logdet_Sigma)
HPFmodel = np.ones((N,1))
norm_HPFmodel = HPFmodel/sigmas_vec[:,None]
HPFparas = lsq_linear(norm_HPFmodel, ravelHPFdata / sigmas_vec).x
print(HPFparas)
# HPFparas, HPFchi2, rank, s = np.linalg.lstsq(norm_HPFmodel,ravelHPFdata / sigmas_vec, rcond=None)
# print(HPFparas)

data_model = np.dot(HPFmodel, HPFparas)
deltachi2 = 0  # chi2ref-np.sum(ravelHPFdata**2)
ravelresiduals = ravelHPFdata - data_model
HPFchi2 = np.nansum((ravelresiduals / sigmas_vec) ** 2)

Npixs_HPFdata = HPFmodel.shape[0]
covphi = HPFchi2 / Npixs_HPFdata * np.linalg.inv(np.dot(norm_HPFmodel.T, norm_HPFmodel))
slogdet_icovphi0 = np.linalg.slogdet(np.dot(norm_HPFmodel.T, norm_HPFmodel))

Npixs_HPFdata = HPFmodel.shape[0]
minus2logL_HPF = Npixs_HPFdata * (1 + np.log(HPFchi2 / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
AIC_HPF = 2 * (HPFmodel.shape[-1]) + minus2logL_HPF
AIC_ref = Npixs_HPFdata * (1 + np.log(np.nansum((ravelHPFdata / sigmas_vec) ** 2) / Npixs_HPFdata) + logdet_Sigma + np.log(2 * np.pi))
print(AIC_HPF,AIC_ref,AIC_ref-AIC_HPF)

print("H1", HPFparas,np.mean(ravelHPFdata))
print(np.sqrt(covphi))
a = np.zeros(np.size(HPFparas))
a[0] = 1
a_err = np.sqrt(lsq_linear(np.dot(norm_HPFmodel.T, norm_HPFmodel) / (HPFchi2 / Npixs_HPFdata), a).x[0])
print(a_err, (HPFchi2 / Npixs_HPFdata))

print("SNR = ",(1-HPFparas[0])/a_err)