import torch
import torch.nn as nn

import numpy as np






def torsionAngleLoss(angles, angles_t, angles_at):
    l_list = []
    angles_n = ()

    for i in range(len(angles)):
        l_list.append(torch.norm(angles[i]))
        angles_n =  angles_n + (angles[i] / l_list[i],)
            
    L_torsion_list = []
    L_anglenorm_list = []

    for i in range(len(angles)):
        L_torsion_f = torch.minimum(torch.pow(torch.norm(angles_n[i] - angles_t[i]),2), torch.norm(angles_n[i] - angles_at[i]))
        L_anglenorm_f = np.abs(l_list[i] - 1)

        L_torsion_list.append(L_torsion_f)
        L_anglenorm_list.append(L_anglenorm_f)
        
    L_torsion = np.mean(L_torsion_list)
    L_anglenorm = np.mean(L_anglenorm_list)

    return L_torsion + 0.02 * L_anglenorm




def computeFAPE(T, x, T_t, x_t, Z = 1e-9, d_clamp = 1e-9, epsilon = 1e-22):
    x = torch.matmul((1/T[0]), x.T).transpose(1,2) - torch.matmul((1/T[0]), T[1].T).transpose(1,2)
    x_t = torch.matmul((1/T_t[0]), x_t.T).transpose(1,2) + torch.matmul((1/T_t[0]), T_t[1].T).transpose(1,2)

    d = np.sqrt(torch.pow(torch.norm(x - x_t), 2).numpy().item() + epsilon)
    L_FAPE = (1 / Z) * np.mean(np.minimum(d_clamp, d))

    return L_FAPE

    
        




