% Code to reproduce the simulation in Figure 5 in the paper 
% "Moving Object Detection Based on Bioinspired Background Subtraction".
% June 2024.
% Written by Zhu'anzhen Zheng and Zhihua Wu.

clear all
close all

% reading stimulus.
obj = VideoReader([pwd,'\artificialStimuli\Jitter.mp4']);
% reading the ground truth.
obj_GT = VideoReader([pwd,'\artificialStimuli\GT_Jitter.mp4']);

%-----preprocessing the visual inputs.-----
Fsize = 12; sigma = 3.5; K = 6;                                  
h = fspecial('gaussian', Fsize, sigma);                   
frame = obj.NumberOfFrames;
patt = zeros(length(1:K:obj.Height), length(1:K:obj.Width), frame);
patt_record = zeros(obj.Height, obj.Width, frame);
GT = zeros(size(patt, 1)-1, size(patt, 2)-1, frame);
for t = 1:frame
      % blurring and downsampling input images to simulate the fly's optics.
      patt_0 = im2double(read(obj, t));
      patt_1 = rgb2gray(patt_0);
      patt_record(:, :, t) = patt_1;
      patt_blur = imfilter(patt_1, h);
      patt(:, :, t) = patt_blur(1:K:end, 1:K:end);
      % preparing the ground truth.
      GT0 = rgb2ind(read(obj_GT, t), 2);
      GT1 = GT0(1:K:end, 1:K:end);      
      GT(:, :, t) = GT1(1:(end-1), 1:(end-1));
end
%-----EMD array module.-----
step = 0.01;                                       % temporal resolution of the EMD array in [s].
% low-pass filter. 
tauL = 0.050;                                      % the LP filter's time constant in [s].  
dL = tauL/step;                       
a_low(1, 1) = 1/(dL+1);
a_low(1, 2)= 1-a_low(1, 1);
% high-pass filter. 
tauH = 0.250;                                      % the HP filter's time constant in [s].                                
dH = tauH/step;                     
b_high(1, 1) = dH/(dH+1);
b_high(1, 2) = b_high(1, 1);
EMD_nx = size(patt, 1);
EMD_ny = size(patt, 2);
Fh = zeros(EMD_nx, EMD_ny);                            
Fd_On = zeros(EMD_nx, EMD_ny);                     
Fd_Off = zeros(EMD_nx, EMD_ny);               
H_On = zeros(EMD_nx-1, EMD_ny-1);                  % output of EMD array in the ON channel along the horizontal axis.
H_Off = zeros(EMD_nx-1, EMD_ny-1);                 % output of EMD array in the OFF channel along the horizontal axis.
V_On = zeros(EMD_nx-1, EMD_ny-1);                  % output of EMD array in the ON channel along the vertical axis.
V_Off = zeros(EMD_nx-1, EMD_ny-1);                 % output of EMD array in the OFF channel along the vertical axis.
EMD_o_H = zeros(EMD_nx-1, EMD_ny-1, frame);        % output of the EMD array in horizontal direction.
EMD_o_V = zeros(EMD_nx-1, EMD_ny-1, frame);        % output of the EMD array in vertical direction.

%-----lobula module, lobula plate (LP) module, and Mo module.-----
Lc_tau = 5;                                        % membrane time constant of all the units in [ms].
Lc_step = 1;                                       % integration time step of all the units in [ms].
% LP units
LP_beta = 0.2;                                     % steepness of the sigmoid activation function of LP units.
LP_Vhalf = -50;                                    % half-activation voltage of the sigmoid activation function of LP units in [mV]. 
Con_EMD_LP = 0.01;                                 % synaptic weight from EMD array to LP units.
LP_v = -50+ones(4, 1);                             % membrane potentials of 4 LP units.
LP_v_record = -50+zeros(4, 1, frame);              % recording membrane potentials of 4 LP units.  
s_w = zeros(4, 1);                                 % blocking coefficients.
% units in the lobula and Mo modules. 
receptive_L = 5;           
RE_L = fspecial('gaussian', receptive_L, receptive_L/6);             % receptive field of units in the Ir, Il, Id, and Iu submodules.
RF_Mo = 3;                                                            
filter_Mo = fspecial('gaussian', RF_Mo, RF_Mo/6);                    % receptive field of units in the Mo submodule.
Con_EMD_Lo = 200;                                                    % synaptic weight from the EMD array to units in Ir, Il, Id, and Iu.
Lo_beta = 0.2;                                                       % steepness of the sigmoid activation function of Ir, Il, Id, Iu, and Mo.
Lo_Vhalf = -40;                                                      % half-activation voltage of the sigmoid activation function of Ir, Il, Id, Iu, and Mo in [mV]. 
Lo_v = -50+ones(5*(EMD_nx-1), (EMD_ny-1));                           % membrane potentials of units in Ir, Il, Id, Iu, and Mo.
Lo_v_record = -50+zeros(5*(EMD_nx-1), (EMD_ny-1), frame);            % reacording membrane potentials of lobula units in Ir, Il, Id, Iu, and Mo.  
Con_r = zeros(EMD_nx-1, EMD_ny-1, frame);                            % output of Ir.
Con_l = zeros(EMD_nx-1, EMD_ny-1, frame);                            % output of Il.          
Con_d = zeros(EMD_nx-1, EMD_ny-1, frame);                            % output of Id.  
Con_u = zeros(EMD_nx-1, EMD_ny-1, frame);                            % output of Iu.
Con_m = zeros(EMD_nx-1, EMD_ny-1, frame);                            % output of Mo.

for t = 2:frame
       % high-pass filtering.  
        Fh = b_high(1, 1)*(patt(:, :, t)-patt(:, :, t-1))+b_high(1, 2)*Fh;    
       % through two parallel half-wave rectifiers.
        On = Rect(1, Fh, 0);
        Off = Rect(-1, Fh, 0); 
       % low-pass filtering.
        Fd_On = a_low(1, 1)*On + a_low(1, 2)*Fd_On; 
        Fd_Off = a_low(1, 1)*Off + a_low(1, 2)*Fd_Off;
       % ON channel. 
        H_On(1:(EMD_nx-1), 1:(EMD_ny-1)) = Fd_On(1:(EMD_nx-1), 1:(EMD_ny-1)).*On(1:(EMD_nx-1), 2:EMD_ny)-On(1:(EMD_nx-1), 1:(EMD_ny-1)).*Fd_On(1:(EMD_nx-1), 2:EMD_ny);
        V_On(1:(EMD_nx-1), 1:(EMD_ny-1)) = Fd_On(1:(EMD_nx-1), 1:(EMD_ny-1)).*On(2:EMD_nx, 1:(EMD_ny-1))-On(1:(EMD_nx-1), 1:(EMD_ny-1)).*Fd_On(2:EMD_nx, 1:(EMD_ny-1));
       % OFF channel. 
        H_Off(1:(EMD_nx-1), 1:(EMD_ny-1)) = Fd_Off(1:(EMD_nx-1), 1:(EMD_ny-1)).*Off(1:(EMD_nx-1), 2:EMD_ny)-Off(1:(EMD_nx-1), 1:(EMD_ny-1)).*Fd_Off(1:(EMD_nx-1), 2:EMD_ny);
        V_Off(1:(EMD_nx-1), 1:(EMD_ny-1)) = Fd_Off(1:(EMD_nx-1), 1:(EMD_ny-1)).*Off(2:EMD_nx, 1:(EMD_ny-1))-Off(1:(EMD_nx-1), 1:(EMD_ny-1)).*Fd_Off(2:EMD_nx, 1:(EMD_ny-1));

        Hon_L = H_On;
        Hon_R = H_On;
        Hoff_L = H_Off;  
        Hoff_R = H_Off;
       % direction-dependent components of the EMD output.
        Hon_L(Hon_L>=0) = 0;
        Hon_R(Hon_R<0) = 0;
        Hoff_L(Hoff_L>=0) = 0;
        Hoff_R(Hoff_R<0) = 0;
        
        Von_U = V_On;
        Von_D = V_On;
        Voff_U = V_Off;  
        Voff_D = V_Off;
        Von_U(Von_U>=0) = 0;
        Von_D(Von_D<0) = 0;
        Voff_U(Voff_U>=0) = 0;
        Voff_D(Voff_D<0) = 0;
        
       % updating the states of 4 LP units by solving the ordinary differential equations.
        Con_LP_R = Con_EMD_LP*(sum(Hon_R(:))+sum(Hoff_R(:)));
        Con_LP_L = -Con_EMD_LP*(sum(Hon_L(:))+sum(Hoff_L(:)));   
        Con_LP_D = Con_EMD_LP*(sum(Von_D(:))+sum(Voff_D(:)));
        Con_LP_U = -Con_EMD_LP*(sum(Von_U(:))+sum(Voff_U(:)));  
        [Con_LP, LP_v_new]  = LP_Units(step, Lc_tau, Lc_step, Con_LP_R, Con_LP_L, Con_LP_D, Con_LP_U, LP_v, LP_Vhalf, LP_beta);  % integrate-and-nonspiking dynamics
        LP_v = LP_v_new;  
        LP_v_record(:, :, t) = LP_v_new;         
        s_w = 1*(Con_LP<=0.9);                      % blocking coefficients determined online depending on the activity states of LP units.
       % updating the states of units in Ir, Il, Id, Iu, and Mo.
        Con_R = conv2(Con_EMD_Lo*(Hon_R + Hoff_R), RE_L, 'same');  
        Con_L = conv2(-Con_EMD_Lo*(Hon_L + Hoff_L), RE_L, 'same');
        Con_D = conv2(Con_EMD_Lo*(Von_D + Voff_D), RE_L, 'same');
        Con_U = conv2(-Con_EMD_Lo*(Von_U + Voff_U), RE_L, 'same');
        [Con_Lc, Lo_v_new] = Lo_Units_Jitter(step, Lc_tau, Lc_step, EMD_nx, Lo_v, Con_R, Con_L, Con_D, Con_U, s_w, filter_Mo, Lo_Vhalf, Lo_beta);
        Lo_v = Lo_v_new;                                          
        Lo_v_record(:, :, t) = Lo_v_new;                
        % recording the output of Ir, Il, Id, Iu, and Mo.
        Con_r(:, :, t) = Con_Lc(1:(EMD_nx-1), :);   
        Con_l(:, :, t) = Con_Lc(EMD_nx:2*(EMD_nx-1), :); 
        Con_d(:, :, t) = Con_Lc((2*(EMD_nx-1)+1):3*(EMD_nx-1), :); 
        Con_u(:, :, t) = Con_Lc((3*(EMD_nx-1)+1):4*(EMD_nx-1), :); 
        Con_m(:, :, t) = Con_Lc((4*(EMD_nx-1)+1):5*(EMD_nx-1), :);
        % Recording output of EMD arrays.
        EMD_o_H(:, :, t) = H_On + H_Off;
        EMD_o_V(:, :, t) = V_On + V_Off;
end

% calculating the F-measures.
F_6 = zeros(6, frame);
confMat = zeros(6, 4);
% Recording output
imBinary_EMDH_record = zeros(size(EMD_o_H));
imBinary_EMDV_record = zeros(size(EMD_o_V));
imBinary_R1_record = zeros(size(Con_r));
imBinary_L1_record = zeros(size(Con_l));
imBinary_D1_record = zeros(size(Con_d));
imBinary_U1_record = zeros(size(Con_u));
imBinary_M1_record = zeros(size(Con_m));

 for i = 1:frame  
      imGT = uint8(255*GT(:, :, i));
     % output of EMD array along the horizontal axis.                   
      imBinary_EMD_01H = mat2gray(EMD_o_H(:, :, i));     
      imBinary_EMD_0H = im2bw(imBinary_EMD_01H, 0.5);        % segmentation threshold was set as a 50% maximum.
      imBinary_EMDH = uint8(255*imBinary_EMD_0H);                
      imBinary_EMDH_record(:, :, i) = imBinary_EMDH;  
     % output of Ir, Il, Id, Iu, and Mo.
      imBinary_EMD_01V = mat2gray(EMD_o_V(:, :, i));     
      imBinary_EMD_0V = im2bw(imBinary_EMD_01V, 0.5);
      imBinary_EMDV = uint8(255*imBinary_EMD_0V); 
      imBinary_EMDV_record(:, :, i) = imBinary_EMDV; 
     
      imBinary_R1_00 = mat2gray(Con_r(:, :, i));
      imBinary_R1_0 = im2bw(imBinary_R1_00, 0.5); 
      imBinary_R1 = uint8(255*imBinary_R1_0);              
      imBinary_R1_record(:, :, i) = imBinary_R1;
     
      imBinary_L1_00 = mat2gray(Con_l(:, :, i));
      imBinary_L1_0 = im2bw(imBinary_L1_00, 0.5); 
      imBinary_L1 = uint8(255*imBinary_L1_0);  
      imBinary_L1_record(:, :, i) = imBinary_L1;     
     
      imBinary_D1_00 = mat2gray(Con_d(:, :, i));
      imBinary_D1_0 = im2bw(imBinary_D1_00, 0.5); 
      imBinary_D1 = uint8(255*imBinary_D1_0); 
      imBinary_D1_record(:, :, i) = imBinary_D1; 
     
      imBinary_U1_00 = mat2gray(Con_u(:, :, i));
      imBinary_U1_0 = im2bw(imBinary_U1_00, 0.5); 
      imBinary_U1 = uint8(255*imBinary_U1_0);
      imBinary_U1_record(:, :, i) = imBinary_U1; 
    
      imBinary_M1_00 = mat2gray(Con_m(:, :, i));
      imBinary_M1_0 = im2bw(imBinary_M1_00, 0.5); 
      imBinary_M1 = uint8(255*imBinary_M1_0); 
      imBinary_M1_record(:, :, i) = imBinary_M1; 
     % calculating the instantaneous F-measures. 
      [F_EMD, confMat(1, :)] = F_measure(imBinary_EMDH, imGT);            % F-measure at the out of EMD array along the horizontal axis.
      [F_Ir, confMat(2, :)] = F_measure(imBinary_R1, imGT);               % F-measure at the output of Ir.
      [F_Il, confMat(3, :)] = F_measure(imBinary_L1, imGT);               % F-measure at the output of Il.
      [F_Id, confMat(4, :)] = F_measure(imBinary_D1, imGT);               % F-measure at the output of Id.
      [F_Iu, confMat(5, :)] = F_measure(imBinary_U1, imGT);               % F-measure at the output of Iu.
      [F_Im, confMat(6, :)] = F_measure(imBinary_M1, imGT);               % F-measure at the output of Mo.
      F_6(:, i) = [F_EMD; F_Ir; F_Il; F_Id; F_Iu; F_Im]; 
end

% obtaining the space-time plot by randomly choosing a row (here 144).
space_time = zeros(frame, size(patt_record, 2));
for j=1:frame       
       space_time(j, :) = patt_record(144, :, j); 
end
%===============================================
figure(1)
set(gcf, 'Units','normalized','Position', [0.05 0.65 0.4 0.2]);
colormap('gray');
% Space-time plot of the stimulus.
h1 = subplot(2, 5, [1 2]);
A = flip(space_time');
imagesc(A, [0, 1]);
hold on;
text(-15, 500, '-90',  'FontName', 'Times New Roman', 'FontSize',7);
text(-12, 50, '90',  'FontName', 'Times New Roman', 'FontSize',7);
text(0, 600, '0.0',  'FontName', 'Times New Roman', 'FontSize',7);
text(100, 600, '1.0',  'FontName', 'Times New Roman', 'FontSize',7);
text(190, 600, '2.0',  'FontName', 'Times New Roman', 'FontSize',7);
text(90, -43, 'Time (s)',  'FontName', 'Times New Roman', 'FontSize',8);
text(-29, 480, 'Space (^o)',  'FontName', 'Times New Roman', 'FontSize',8, 'rotation', 90);
set(gca, 'XTick', 1:45:91, 'XTickLabel', {''},  'FontName', 'Times New Roman', 'FontSize',8); 
set(gca, 'YTick', 1:23:46, 'YTickLabel', {''},  'FontName', 'Times New Roman', 'FontSize',8); 

% Time courses of figure velocity components. 1 pixel/s = 0.33 deg/s.
h2 = subplot(2, 5, [6 7]);
stimulus_velocity_FG = [1, 2, 33, 34, 65, 66, 99, 100, 165, 166, 200; 0, -300, -300, 300, 300, -400, -400, -300, -300, -200, -200]; % pixels/s
stimulus_velocity_BG = [1, 2, 200; 0, 300, 300];    % pixels/s
plot(stimulus_velocity_FG(1, :), stimulus_velocity_FG(2, :), '-', 'color', [214/255  62/255  50/255]); 
hold on;
plot(stimulus_velocity_BG(1, :), stimulus_velocity_BG(2, :), '-', 'color', [0  0  0]);
Mi = 140;             % the time instant the snapshots were taken.
plot([Mi, Mi], [-600, 600], ':', 'color', [0.5, 0.5, 0.5], 'Linewidth', 1);
text(10, 460, '(in horizontal)',  'FontName', 'Times New Roman', 'FontSize',7);
text(160, 460, 'background',  'FontName', 'Times New Roman', 'FontSize',7);
text(180, -40, 'object', 'color', [214/255  62/255  50/255],  'FontName', 'Times New Roman', 'FontSize',7);
set(gca, 'XLim', [0, 200], 'XTick', 0:50:200, 'XTickLabel', {'0.0', '0.5', '1.0', '1.5', '2.0'},   'FontName', 'Times New Roman', 'FontSize',7);
set(gca, 'YLim', [-600, 600], 'YTick', -600:300:600, 'YTickLabel', {'-198', '-99', '0', '99', '198'},  'FontName', 'Times New Roman', 'FontSize',7);
set(gca,'TickLength', 1.2*get(gca,'ticklength'));
box off;
ylabel('Velocity (^o/s)',  'FontName', 'Times New Roman', 'FontSize',8);  

% Response snapshots of EMD arrays.
h3 = subplot(2, 5, 4);
imagesc(EMD_o_H(:, :, Mi), [-0.09, 0.09]);
text(19, -14, 'Output of the ON+OFF EMD array', 'FontName', 'Times New Roman', 'FontSize', 8);
text(2, -4, '(in horizontal direction)', 'FontName', 'Times New Roman', 'FontSize', 7);
text(-15, 2, '45^o', 'FontName', 'Times New Roman', 'FontSize', 7); 
text(-18, 38, '-45^o', 'FontName', 'Times New Roman', 'FontSize', 7); 
set(gca, 'XTick', 0:90:90, 'XTickLabel', {  });
set(gca, 'YTick', 0:45:45, 'YTickLabel', {  });  

h4 = subplot(2, 5, 5);
imagesc(EMD_o_V(:, :, Mi), [-0.09, 0.09]);
text(7, -4, '(in vertical direction)', 'FontName', 'Times New Roman', 'FontSize', 7);
set(gca, 'XTick', 0:90:90, 'XTickLabel', {  });
set(gca, 'YTick', 0:45:45, 'YTickLabel', {  });  
bar1 = colorbar('YTick', -0.09:0.09:0.09, 'YTickLabel', {'-0.09', '', '+0.09'}, 'FontName', 'Times New Roman', 'FontSize',8); 
set(bar1, 'Position',[0.8445    0.55   0.01    0.2]);

h5 = subplot(2, 5, 9);
imagesc(imBinary_EMDH_record(:, :, Mi), [0, 1]); 
hold on;
[fy, fx] = find(GT(:, :, Mi), 1, 'first');
[Ly, Lx] = find(GT(:, :, Mi), 1, 'last');
plot([fx  fx], [fy Ly], '-r');
hold on;
plot([Lx  Lx], [fy Ly], '-r');
hold on;
plot([fx  Lx], [fy fy], '-r');
hold on;
plot([fx  Lx], [Ly Ly], '-r');
hold on; 
set(gca, 'XTick', 0:90:90, 'XTickLabel', {  });
set(gca, 'YTick', 0:45:45, 'YTickLabel', {  });  
text(-15, 2, '45^o', 'FontName', 'Times New Roman', 'FontSize', 7); 
text(-18, 38, '-45^o', 'FontName', 'Times New Roman', 'FontSize', 7); 
text(-2, 49, '-90^o', 'FontName', 'Times New Roman', 'FontSize', 7);  
text(78, 49, '90^o', 'FontName', 'Times New Roman', 'FontSize', 7);  

h6 = subplot(2, 5, 10);
imagesc(imBinary_EMDV_record(:, :, Mi), [0, 1]);
hold on;
plot([fx  fx], [fy Ly], '-r');
hold on;
plot([Lx  Lx], [fy Ly], '-r');
hold on;
plot([fx  Lx], [fy fy], '-r');
hold on;
plot([fx  Lx], [Ly Ly], '-r');
hold on; 
set(gca, 'XTick', 0:90:90, 'XTickLabel', {  },  'FontName', 'Times New Roman', 'FontSize',8);
set(gca, 'YTick', 0:45:45, 'YTickLabel', {  },  'FontName', 'Times New Roman', 'FontSize',8); 
set(h1, 'position', [0.1600    0.55    0.2866    0.3]);
set(h2, 'position', [0.1600    0.15    0.2866    0.3]);
set(h3, 'position', [0.5584    0.55  0.1337    0.3]);
set(h4, 'position', [0.7013    0.55   0.1337    0.3]);
set(h5, 'position', [ 0.5584    0.15    0.1337    0.3]);
set(h6, 'position', [0.7013   0.15  0.1337  0.3]);
%--------------------------------------------------------
figure(2)
set(gcf, 'Units','normalized','Position', [0.05 0.45 0.5 0.1]);
colormap('parula'); 
% Instantaneous activity of units in Ir, Il, Id, Iu, and Mo.
nx_EMD = size(EMD_o_H, 1);
A = {Lo_v_record(1:nx_EMD, :,:), Lo_v_record((nx_EMD+1):2*nx_EMD, :,  :), ...
       Lo_v_record((2*nx_EMD+1):3*nx_EMD, :, :), Lo_v_record((3*nx_EMD+1):4*nx_EMD, :, :), Lo_v_record((4*nx_EMD+1):5*nx_EMD-1, :, :)};
A_title = {'Ir', 'Il', 'Id', 'Iu', 'Mo'};
for i=1:5
     subplot(1, 5, i)
     for j = 1:frame
         imagesc(A{i}(:, :, j), [-74, -10]);
         axis off;
         pause(0.01);
     end
     title(A_title{i});
end 

bar2 = colorbar('YTick', -74:32:-10,  'YTickLabel', {'-74', '-42', '-10'}, 'FontSize',8); 
colormap(bar2,'parula');
set(bar2, 'Position',[0.94    0.22   0.01    0.35]);
%--------------------------------------------------------
figure(3)
set(gcf, 'Units','normalized','Position', [0.05 0.25 0.5 0.1]);
colormap('gray');
B = {imBinary_R1_record, imBinary_L1_record, imBinary_D1_record, imBinary_U1_record, imBinary_M1_record};
B_title = {'Ir', 'Il', 'Id', 'Iu', 'Mo'};

for i=1:5
     subplot(1, 5, i)
     for j = 1:frame
         imagesc(B{i}(:, :, j));
         axis off;
         pause(0.01);
     end
     title(B_title{i});
end 

bar3 = colorbar('YTick', 0:255:255,  'YTickLabel', {'0', '1'});
set(bar3, 'Position',[0.94    0.22   0.01    0.35]);
%-------------------------------------------------------
figure(4)
set(gcf, 'Units','normalized','Position', [0.05 0.10 0.4 0.2]);
subplot(1, 5, [1, 2])
% Membrane potentials of the 4 LP units .
Vm_LP(:, 1) = LP_v_record(1, 1, :);
Vm_LP(:, 2) = LP_v_record(2, 1, :);
Vm_LP(:, 3) = LP_v_record(3, 1, :);
Vm_LP(:, 4) = LP_v_record(4, 1, :);

plot(1:frame, Vm_LP(:, 1), ':', 'color', [30/255, 80/255, 173/255], 'Linewidth', 1);
hold on;
plot(1:frame, Vm_LP(:, 2), '-', 'color', [30/255, 80/255, 173/255]);
plot(1:frame, Vm_LP(:, 3), ':', 'color', [220/255, 140/255, 0/255], 'Linewidth', 1);
plot(1:frame, Vm_LP(:, 4), '-', 'color', [220/255, 140/255, 0/255]);

% V_thr is the membrane potential corresponding to the threshold Sthr.
V_thr = 0.2*log(9)-50;
plot([1, frame], [V_thr, V_thr], '-', 'color', [160/255, 160/255, 160/255]);
box off;
text(-40, -58, 'Vm of LPTCs units (mV)',  'FontName', 'Times New Roman', 'FontSize',8, 'Rotation', 90);
text(74, -66, 'Time (s)',  'FontName', 'Times New Roman', 'Fontsize', 8);
set(gca, 'XLim', [0, 200], 'XTick', 0:50:200, 'XTickLabel', {'0.0', '0.5', '1.0', '1.5', '2.0'},  'FontName', 'Times New Roman', 'FontSize',8);
set(gca, 'YLim', [-60, -30], 'YTick', -60:10:-30, 'YTickLabel', {'-60', '-50', '-40', '-30'},  'FontName', 'Times New Roman', 'FontSize',8);
set(gca, 'position', [0.1500    0.18    0.19    0.65]);

subplot(1, 5, 4)
% instantaneous F-measures.
plot(1:frame, F_6(2, :), ':', 'color', [30/255, 80/255, 173/255], 'Linewidth', 1);
hold on;
plot(1:frame, F_6(3, :), '-', 'color', [30/255, 80/255, 173/255]);
plot(1:frame, F_6(4, :), ':', 'color', [220/255, 140/255, 0/255], 'Linewidth', 1);
plot(1:frame, F_6(5, :), '-', 'color', [220/255, 140/255, 0/255]);
box off;
set(gca, 'XLim', [0, 200], 'XTick', 0:50:200, 'XTickLabel', {'0.0', '0.5', '1.0', '1.5', '2.0'},  'FontName', 'Times New Roman', 'FontSize',8);
set(gca, 'YLim', [0, 1.0], 'YTick', 0:0.5:1.0, 'YTickLabel', {'0.0', '0.5', '1.0'},  'FontName', 'Times New Roman', 'FontSize',8);
text(74, -0.2, 'Time (s)',  'FontName', 'Times New Roman', 'Fontsize', 8);
ylabel('F-measure', 'FontSize',8); 
set(gca, 'position', [0.4    0.18    0.19    0.65]);

subplot(1, 5, 5)
plot(1:frame, F_6(1,:), '-', 'color', [0 0.4 0]);
hold on;
plot(1:frame, F_6(6,:), '-', 'color', [150/255  0/255  39/255]);
box off;
text(74, -0.2, 'Time (s)',  'FontName', 'Times New Roman', 'Fontsize', 8);
set(gca, 'XLim', [0, 200], 'XTick', 0:50:200, 'XTickLabel', {'0.0', '0.5', '1.0', '1.5', '2.0'},  'FontSize',8);
set(gca, 'YLim', [0, 1.0], 'YTick', 0:0.5:1.0, 'YTickLabel', {'0.0', '0.5', '1.0'},  'FontName', 'Times New Roman', 'FontSize',8);
set(gca, 'position', [0.64    0.18   0.19    0.65]);
%====================================================
