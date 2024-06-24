% Solving differential equations by using the 4th order Runge-Kutta method. 

function  v1 = RK4(h, v0, I, tau_m)
% h0: integration time step in [ms].
% tau_m: membrane time constant of units in [ms].

K1=h*ydot1(v0, I, tau_m);
K2=h*ydot1(v0+0.5*K1, I, tau_m);
K3=h*ydot1(v0+0.5*K2, I, tau_m);
K4=h*ydot1(v0+K3, I, tau_m);
v1=v0+(K1 + 2*(K2 +K3) +K4)/6;

end

function  vd = ydot1(v, I, tau_m)
% integrate-and-nonspiking neuron model
EL = -50;                   % resting membrane potential in [mV].
vd =(EL-v+I)/tau_m;

end



