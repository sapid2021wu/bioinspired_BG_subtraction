% integration of the LP module for step/Lc_step time steps within one EMD step.

function  [Con_LP, LP_v_new] = LP_Units(step, Lc_tau, Lc_step, Con_LP_R, Con_LP_L, Con_LP_D, Con_LP_U, LP_v, LP_Vhalf, LP_beta)
% step: temporal step of the EMD array in [s]. 
% Lc_tau: membrane time constant of the LP units in [ms].             
% Lc_step: integration step of the LP units in [ms].
% Con_LP_R, Con_LP_L, Con_LP_D, Con_LP_U: input conductances. 
% LP_v, LP_v_new: membrane potentials of the LP units in [mV].  
% LP_beta: steepness of the activation function.
% LP_Vhalf: half-activation voltage of the activation function in [mV].
% Con_LP: output of the LP units.

E_e = 0;              % reversal potential of excitatory synapses in [mV].
E_i = -80;            % reversal potential of inhibitory synapses in [mV].
        
for t_Lc=1:Lc_step:(1000*step)
    Con_LP = LP_v;    
    Con_LP = 1./(1+exp((LP_Vhalf-Con_LP)/LP_beta)); 
    
    I(1, 1) = Con_LP_R*(E_e-LP_v(1, 1)) + Con_LP_L*0.5*(E_i-LP_v(1, 1));
    I(2, 1) = Con_LP_L*(E_e-LP_v(2, 1)) + Con_LP_R*0.5*(E_i-LP_v(2, 1));
    I(3, 1) = Con_LP_D*(E_e-LP_v(3, 1)) + Con_LP_U*0.5*(E_i-LP_v(3, 1));
    I(4, 1) = Con_LP_U*(E_e-LP_v(4, 1)) + Con_LP_D*0.5*(E_i-LP_v(4, 1));

    LP_v_new = RK4(Lc_step, LP_v, I, Lc_tau);
    LP_v = LP_v_new;                                                         
end
 
end
