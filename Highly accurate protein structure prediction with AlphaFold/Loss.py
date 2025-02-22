import torch
import torch.nn as nn

import numpy as np




class torsionAngleLoss(nn.Module):
    def __init__(self,):
        super(torsionAngleLoss, self).__init__()

    def forward(self, angles, angles_t, angles_at):
        l_list = []
        for i in range(len(angles)):
            l_list[i] = torch.norm(angles[i])
            angles[i] = angles[i] / l_list[i]
            
        L_torsion_list = []
        L_anglenorm_list = []

        for i in range(len(angles)):
            L_torsion_f = torch.min(torch.pow(torch.norm(angles[i] - angles_t[i]),2), torch.norm(angles[i] - angles_at[i]))
            L_anglenorm_f = np.abs(l_list[i] - 1)

            L_torsion_list.append(L_torsion_f)
            L_anglenorm_list.append(L_anglenorm_f)
        
        L_torsion = np.mean(L_torsion_list)
        L_anglenorm = np.mean(L_anglenorm_list)

        return L_torsion + 0.02 * L_anglenorm




class computeFAPE(nn.Module):
    def __init__(self,):
        super(computeFAPE, self).__init__()

    def forward(self,):
        pass




