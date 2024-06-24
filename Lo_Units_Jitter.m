% integration of the Ir, Il, Id, Iu, and Mo submodules for step/Lc_step time steps within one EMD step.

function  [Con_Lc, Lo_v_new] = Lo_Units_Jitter(step, Lc_tau, Lc_step, EMD_nx, Lo_v, Con_R, Con_L, Con_D, Con_U, s_w, filter_Mo, Lo_Vhalf, Lo_beta)
% step: temporal step of the EMD array in [s]. 
% Lc_tau: membrane time constant of Ir, Il, Id, Iu, and Mo in [ms].             
% Lc_step: integration step of Ir, Il, Id, Iu, and Mo in [ms].
% Lo_v and Lo_v_new: membrane potentials of Ir, Il, Id, Iu, and Mo in [mV].  
% Lo_beta: steepness of the activation function.
% Lo_Vhalf: half-activation voltage of the activation function in [mV].
% Con_Lc: output of Ir, Il, Id, Iu, and Mo.

E_e = 0;               % reversal potential of excitatory synapses in [mV].
E_i = -80;             % reversal potential of inhibitory synapses in [mV].

for t_Lc=1:Lc_step:(1000*step)
    Con_Lc = Lo_v;          
    Con_Lc = 1./(1+exp((Lo_Vhalf-Con_Lc)/Lo_beta));          
    Con_m = conv2((s_w(1, :)*Con_Lc(1:(EMD_nx-1), :) + s_w(2, :)*Con_Lc(EMD_nx:2*(EMD_nx-1), :)+ ...
              s_w(3, :)*Con_Lc((2*(EMD_nx-1)+1):3*(EMD_nx-1), :)+ s_w(4, :)*Con_Lc((3*(EMD_nx-1)+1):4*(EMD_nx-1), :)), filter_Mo, 'same');  
                                              
    I_1(1:(EMD_nx-1), :) = Con_R.*(E_e-Lo_v(1:(EMD_nx-1), :)) + Con_L.*(E_i-Lo_v(1:(EMD_nx-1), :));                                                        % input to Ir.
    I_1(EMD_nx:2*(EMD_nx-1), :) = Con_L.*(E_e-Lo_v(EMD_nx:2*(EMD_nx-1), :)) + Con_R.*(E_i-Lo_v(EMD_nx:2*(EMD_nx-1), :));                                   % input to Il.
    I_1((2*(EMD_nx-1)+1):3*(EMD_nx-1), :) = Con_D.*(E_e-Lo_v((2*(EMD_nx-1)+1):3*(EMD_nx-1), :)) + Con_U.*(E_i-Lo_v((2*(EMD_nx-1)+1):3*(EMD_nx-1), :));     % input to Id.
    I_1((3*(EMD_nx-1)+1):4*(EMD_nx-1), :) = Con_U.*(E_e-Lo_v((3*(EMD_nx-1)+1):4*(EMD_nx-1), :)) + Con_D.*(E_i-Lo_v((3*(EMD_nx-1)+1):4*(EMD_nx-1), :));     % input to Iu.
    I_1((4*(EMD_nx-1)+1):5*(EMD_nx-1), :) = Con_m.*(E_e-Lo_v((4*(EMD_nx-1)+1):5*(EMD_nx-1), :));                                                           % input to Im.
   
    Lo_v_new = RK4(Lc_step, Lo_v, I_1, Lc_tau);
    Lo_v = Lo_v_new;                                                 
end
 
end
