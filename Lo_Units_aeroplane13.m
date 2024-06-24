function  [Con_Lc, Lo_v_new] = Lo_Units_aeroplane13(step, Lc_tau, Lc_step, EMD_nx, Lo_v, Con_R, Con_L, Con_D, Con_U,  s_w, filter_Mo, Lo_Vhalf, Lo_beta)
% step: EMD step in [s].
% Con_Lc: output of lobula units. 
% Lo_v, Lo_v_new: membrane potentials of the lobula units.  
% Lc_tau: membrane time constant of lobula units in [ms].             
% Lc_step: integration step of the lobula units in [ms].
% Lo_beta: steepness of the activation function.
% Lo_Vhalf: half-activation voltage of the activation function in [mV].

E_e = 0;               % reversal potential of excitatory synapses in [mV].
E_i = -80;            % reversal potential of inhibitory synapses in [mV].

for t_Lc=1:Lc_step:(1000*step)                      
    Con_Lc = Lo_v;          
    Con_Lc = 1./(1+exp((Lo_Vhalf-Con_Lc)/Lo_beta));      
    Con_m = conv2((s_w(1, :)*Con_Lc(1:EMD_nx, :) + s_w(2, :)*Con_Lc((EMD_nx+1):2*EMD_nx, :)+ s_w(3, :)*Con_Lc((2*EMD_nx+1):3*EMD_nx, :)+ s_w(4, :)*Con_Lc((3*EMD_nx+1):4*EMD_nx, :)), filter_Mo, 'same');  
                                              
    I_1(1:EMD_nx, :) = Con_R.*(E_e-Lo_v(1:EMD_nx, :)) + Con_L.*(E_i-Lo_v(1:EMD_nx, :));                                          
    I_1((EMD_nx+1):2*EMD_nx, :) = Con_L.*(E_e-Lo_v((EMD_nx+1):2*EMD_nx, :)) + Con_R.*(E_i-Lo_v((EMD_nx+1):2*EMD_nx, :));    
    I_1((2*EMD_nx+1):3*EMD_nx, :) = Con_D.*(E_e-Lo_v((2*EMD_nx+1):3*EMD_nx, :)) + Con_U.*(E_i-Lo_v((2*EMD_nx+1):3*EMD_nx, :)); 
    I_1((3*EMD_nx+1):4*EMD_nx, :) = Con_U.*(E_e-Lo_v((3*EMD_nx+1):4*EMD_nx, :)) + Con_D.*(E_i-Lo_v((3*EMD_nx+1):4*EMD_nx, :));
    I_1((4*EMD_nx+1):5*EMD_nx, :) = Con_m.*(E_e-Lo_v((4*EMD_nx+1):5*EMD_nx, :));                                                                  
        
    Lo_v_new = RK4(Lc_step, Lo_v, I_1, Lc_tau);
    Lo_v = Lo_v_new;                                                     
end
 
end