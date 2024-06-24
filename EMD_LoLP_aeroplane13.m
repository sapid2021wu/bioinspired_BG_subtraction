% Code to reproduce the simulation in Figure 7 (with aeroplane13.avi as the stimulus )  
% in the paper "Moving Object Detection Based on Bioinspired Background Subtraction".
% June 2024.
% Written by Zhu'anzhen Zheng and Zhihua Wu.

close  all
clear  all

% reading the video images
videoSetName = 'aeroplane13';
obj = VideoReader([pwd,'\Dataset\',videoSetName,'.avi']);
%-----preprocessing the visual inputs.-----
frame = 101;
frame_begin	= 99;                % start frame number.
input_original = uint8(zeros(obj.Height, obj.Width, 3, frame));
frame_size0 = zeros(obj.Height, obj.Width);
patt = zeros(size(imresize(frame_size0, 0.2), 1), size(imresize(frame_size0, 0.2), 2), frame);
imGT = zeros(obj.Height, obj.Width, frame);

for t = 1:frame
    % preparing the stimulus input.
    input_original(:,:,:,t) = read(obj, frame_begin+t);
    input_image = double(read(obj, frame_begin+t));
    if size(input_image, 3) > 1
        input_image = input_image(:,:, 2)/255;
    end
    patt(:, :, t) = imresize(input_image, 0.2);  
    % preparing the ground truth.
    name1 = [pwd, '\Dataset\Labels\' videoSetName, '_', num2str(frame_begin+t-1, '%05d'), '.txt'];
    fid1 = fopen(name1, 'r');
    a1 = textscan(fid1, '%s %d %d %d %d', 'Delimiter',' ');
    no_GT = size(a1{1}, 1);
    fclose(fid1);
    for i = 1:no_GT
        imGT(a1{3}(i,1):a1{5}(i,1), a1{2}(i,1):a1{4}(i,1), t) = 1;
    end
    
end
%-----EMD array module.-----
step = 0.01;                                      % temporal resolution of the EMD arrays in [s].
% low-pass filtering.
tauL = 0.050;                                     % the low-pass filter's time constant in [s].
dL = tauL/step;
a_low(1, 1) = 1/(dL+1);
a_low(1, 2)= 1-a_low(1, 1);
% high-pass filtering.
tauH = 0.250;                                     % the high-pass filter's time constant in [s].
dH = tauH/step;
b_high(1, 1) = dH/(dH+1);
b_high(1, 2) = b_high(1, 1);
EMD_nx = size(patt, 1);
EMD_ny = size(patt, 2);
Fh = zeros(EMD_nx, EMD_ny);                       
Fd_On = zeros(EMD_nx, EMD_ny);               
Fd_Off = zeros(EMD_nx, EMD_ny); 
H_On = zeros(EMD_nx, EMD_ny);                     % output of EMD array in the ON channel along the horizontal axis.
H_Off = zeros(EMD_nx, EMD_ny);                    % output of EMD array in the OFF channel along the horizontal axis.
V_On = zeros(EMD_nx, EMD_ny);                     % output of EMD array in the ON channel along the vertical axis.
V_Off = zeros(EMD_nx, EMD_ny);                    % output of EMD array in the OFF channel along the vertical axis.
EMD_o_H = zeros(obj.Height, obj.Width, frame);    % output of the EMD array in horizontal direction.
EMD_o_V = zeros(obj.Height, obj.Width, frame);    % output of the EMD array in vertical direction.
% -----lobula module, lobula plate (LP) module, and Mo module.-----
Lc_tau = 5;                                       % membrane time constant of all the units in [ms].
Lc_step = 1;                                      % integration time step of all the units in [ms].
% LP units
LP_beta = 0.2;                                    % steepness of the sigmoid activation function of LP units.
LP_Vhalf = -50;                                   % half-activation voltage of the sigmoid activation function of LP units in [mV].
Con_EMD_LP = 0.003;                               % synaptic weight from EMD array to LP units.
LP_v = -50+ones(4, 1);                            % membrane potentials of 4 LP units.
LP_v_1 = -50+zeros(4, 1, frame);                  % recording membrane potentials of 4 LP units.
s_w = zeros(4, 1);                                % blocking coefficients.
% units in the lobula and Mo modules. 
receptive_s = 17;                                               
RF_L = 1.0*fspecial('gaussian', receptive_s, receptive_s/6);    % receptive field of units in the Ir, Il, Id, and Iu submodules.
receptive_Mo = 3;
filter_Mo = fspecial('gaussian', receptive_Mo, receptive_Mo/6); % receptive field of units in the Mo submodule.
Con_EMD_Lo = 200;                                               % synaptic weight from the EMD array to units in Ir, Il, Id, and Iu.
Lo_beta = 0.2;                                                  % steepness of the sigmoid activation function of Ir, Il, Id, Iu, and Mo.
Lo_Vhalf = -40;                                                 % half-activation voltage of the sigmoid activation function of Ir, Il, Id, Iu, and Mo in [mV].                                               
Lo_v = -50+ones(5*EMD_nx, EMD_ny);                              % membrane potentials of units in Ir, Il, Id, Iu, and Mo.
Con_r = zeros(obj.Height, obj.Width, frame);                    % output of Ir.
Con_l = zeros(obj.Height, obj.Width, frame);                    % output of Il.
Con_d = zeros(obj.Height, obj.Width, frame);                    % output of Id.
Con_u = zeros(obj.Height, obj.Width, frame);                    % output of Iu.
Con_m = zeros(obj.Height, obj.Width, frame);                    % output of Mo.

for t = 2:frame
    % high-pass filtering.  
     Fh = b_high(1, 1)*(patt(:, :, t)-patt(:, :, t-1))+b_high(1, 2)*Fh;
    % through two parallel half-wave rectifiers.
     On = Rect(1, Fh, 0);      
     Off = Rect(-1, Fh, 0);
    % low-pass filtering.
     Fd_On = a_low(1, 1)*On+a_low(1, 2)*Fd_On;     
     Fd_Off = a_low(1, 1)*Off+a_low(1, 2)*Fd_Off;
    % ON pathway
     H_On(1:(EMD_nx-1), 1:(EMD_ny-1)) = Fd_On(1:(EMD_nx-1), 1:(EMD_ny-1)).*On(1:(EMD_nx-1), 2:EMD_ny)-On(1:(EMD_nx-1), 1:(EMD_ny-1)).*Fd_On(1:(EMD_nx-1), 2:EMD_ny);
     V_On(1:(EMD_nx-1), 1:(EMD_ny-1)) = Fd_On(1:(EMD_nx-1), 1:(EMD_ny-1)).*On(2:EMD_nx, 1:(EMD_ny-1))-On(1:(EMD_nx-1), 1:(EMD_ny-1)).*Fd_On(2:EMD_nx, 1:(EMD_ny-1));
    % OFF pathway
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
     [Con_LP, LP_v_new] = LP_Units(step, Lc_tau, Lc_step, Con_LP_R, Con_LP_L, Con_LP_D, Con_LP_U, LP_v, LP_Vhalf, LP_beta);
     LP_v = LP_v_new;     
     LP_v_1(:, :, t) = LP_v_new;
     s_w = 1*(Con_LP<=0.9);      % blocking coefficients determined online depending on the activity states of LP units.
   %  updating the states of units in Ir, Il, Id, Iu, and Mo.
     Con_R = conv2(Con_EMD_Lo*(Hon_R + Hoff_R), RF_L, 'same');     
     Con_L = conv2(-Con_EMD_Lo*(Hon_L + Hoff_L), RF_L, 'same');    
     Con_D = conv2(Con_EMD_Lo*(Von_D + Voff_D), RF_L, 'same');     
     Con_U = conv2(-Con_EMD_Lo*(Von_U + Voff_U), RF_L, 'same');    
     [Con_Lc, v_new] = Lo_Units_aeroplane13(step, Lc_tau, Lc_step, EMD_nx, Lo_v, Con_R, Con_L, Con_D, Con_U, s_w, filter_Mo, Lo_Vhalf, Lo_beta);
     Lo_v = v_new;
   % restoring the output of EMD arrays, Ir, Il, Id, Iu, and Mo to original input size.
     EMD_o_H(:, :, t) = imresize(H_On + H_Off, [obj.Height, obj.Width]);
     EMD_o_V(:, :, t) = imresize(V_On + V_Off, [obj.Height, obj.Width]);
     Con_r(:, :, t) = imresize(Con_Lc(1:EMD_nx, :), [obj.Height, obj.Width], 'bilinear');
     Con_l(:, :, t) = imresize(Con_Lc((EMD_nx+1):2*EMD_nx, :), [obj.Height, obj.Width], 'bilinear');
     Con_d(:, :, t) = imresize(Con_Lc((2*EMD_nx+1):3*EMD_nx, :), [obj.Height, obj.Width], 'bilinear');
     Con_u(:, :, t) = imresize(Con_Lc((3*EMD_nx+1):4*EMD_nx, :), [obj.Height, obj.Width], 'bilinear');
     Con_m(:, :, t) = imresize(Con_Lc((4*EMD_nx+1):5*EMD_nx, :), [obj.Height, obj.Width], 'bilinear');

end

% calculating the F-measures.
confusionMatrix = zeros(6, 4);
confMat = zeros(6, 4);
F_6 = zeros(6, frame);
% Recording output.
imBinary_EMD_record = zeros(size(EMD_o_H));
imBinary_R1_record = zeros(size(EMD_o_H));
imBinary_L1_record = zeros(size(EMD_o_H));
imBinary_D1_record = zeros(size(EMD_o_H));
imBinary_U1_record = zeros(size(EMD_o_H));
imBinary_M1_record = zeros(size(EMD_o_H));

for i = 2:frame
    imGT1 = uint8(255*imGT(:, :, i));
    % output of the EMD_o_H module.
    imBinary_EMD_01 = mat2gray(EMD_o_H(:, :, i));
    imBinary_EMD_0 = im2bw(imBinary_EMD_01, 0.5);         % segmentation threshold was set as a 50% maximum.
    imBinary_EMD = uint8(255*imBinary_EMD_0);                
    imBinary_EMD_record(:, :, i) = imBinary_EMD;
    % output of the Ir, Il, Id, Iu, and Mo submodules.
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
    [F_EMD, confMat(1, :)] = F_measure(imBinary_EMD, imGT1);
    [F_Ir, confMat(2, :)] = F_measure(imBinary_R1, imGT1);
    [F_Il, confMat(3, :)] = F_measure(imBinary_L1, imGT1);
    [F_Id, confMat(4, :)] = F_measure(imBinary_D1, imGT1);
    [F_Iu, confMat(5, :)] = F_measure(imBinary_U1, imGT1);
    [F_Mo, confMat(6, :)] = F_measure(imBinary_M1, imGT1);
    F_6(:, i) = [F_EMD; F_Ir; F_Il; F_Id; F_Iu; F_Mo];
    % calculating TP, FP, FN, and TN quantities
    confusionMatrix(1, :) = confusionMatrix(1, :) + confMat(1, :);
    confusionMatrix(2, :) = confusionMatrix(2, :) + confMat(2, :);
    confusionMatrix(3, :) = confusionMatrix(3, :) + confMat(3, :);
    confusionMatrix(4, :) = confusionMatrix(4, :) + confMat(4, :);
    confusionMatrix(5, :) = confusionMatrix(5, :) + confMat(5, :);
    confusionMatrix(6, :) = confusionMatrix(6, :) + confMat(6, :);
end
% calculating the average the F-measures. 
F_average = zeros(1, 6);
for i=1:6
    F_average(1, i) = 2.0*confusionMatrix(i, 1)/(2.0*confusionMatrix(i, 1) + confusionMatrix(i, 2) + confusionMatrix(i, 3));
end

%===================================================================
figure(1)
colormap('gray');
set(gcf, 'Units', 'normalized', 'Position', [0.05 0.25 0.4 0.36]);

subplot(3, 4, 1)
for j = 1:frame
    imagesc(input_original(:,:,:,j));
    axis off;
    text(20, 30, num2str(j), 'color', [1, 1, 1]);
    pause(0.01);
end
title('Input images');

subplot(3, 4, 2)
for j = 1:frame
    imagesc(imGT(:, :, j));
    axis off;
    text(20, 30, num2str(j), 'color', [1, 1, 1]);
    pause(0.01);
end
title('Groundtruth');

subplot(3, 4, 3)
for j = 1:frame
    imagesc(imBinary_EMD_record(:, :, j));
    axis off;
    text(20, 30, num2str(j), 'color', [1, 1, 1]);
    pause(0.01);
end
title({'Output of the ON+OFF EMD array' '(in horizontal direction)'});

A = {imBinary_R1_record, imBinary_L1_record, imBinary_D1_record, imBinary_U1_record, imBinary_M1_record};
A_title = {'Output of Ir', 'Output of Il', 'Output of Id', 'Output of Iu', 'Output of Mo'};
for i=1:5
      subplot(3, 4, i+4)
for j = 1:frame
      imagesc(A{i}(:, :, j));
      axis off;
      text(20, 30, num2str(j), 'color', [1, 1, 1]);
      pause(0.01);
end
      title(A_title{i});
end
%============================================
figure(2)
set(gcf, 'Units','normalized','Position', [0.05 0.25 0.5 0.2]);

subplot(1, 4, 1)
% Membrane potentials of the 4 LP units.
Vm_LP(:, 1) = LP_v_1(1, 1, :);
Vm_LP(:, 2) = LP_v_1(2, 1, :);
Vm_LP(:, 3) = LP_v_1(3, 1, :); 
Vm_LP(:, 4) = LP_v_1(4, 1, :);
plot(1:frame, Vm_LP(:, 1),  '--', 'color', [30/255, 80/255, 173/255], 'Linewidth', 1);
hold on;
plot(1:frame, Vm_LP(:, 2), '-', 'color', [30/255, 80/255, 173/255]);
plot(1:frame, Vm_LP(:, 3), '--', 'color', [220/255, 140/255, 0/255], 'Linewidth', 1);
plot(1:frame, Vm_LP(:, 4), '-', 'color', [220/255, 140/255, 0/255]);
% V_thr is the membrane potential corresponding to the threshold Sthr.
V_thr = 0.2*log(9)-50;
plot([1, frame], [V_thr, V_thr], '-', 'color', [160/255, 160/255, 160/255]);
set(gca, 'XLim', [0, 100]);
set(gca, 'YLim', [-52, -48], 'YTick', -52:2:-48, 'YTickLabel', {'-52', '-50','-48'});
xlabel('Time (s)');
ylabel('Vm (mV)');  
box off;

subplot(1, 4, 2)
% instantaneous F-measures.
plot(1:frame, F_6(2, :), '--', 'color', [30/255, 80/255, 173/255], 'Linewidth', 1);
hold on;
plot(1:frame, F_6(3, :), '-', 'color', [30/255, 80/255, 173/255]);
plot(1:frame, F_6(4, :), '--', 'color', [220/255, 140/255, 0/255], 'Linewidth', 1);
plot(1:frame, F_6(5, :),   '-', 'color', [220/255, 140/255, 0/255]);
set(gca, 'XLim', [0, frame], 'YLim', [0, 1.0]);
xlabel('Time (s)'); 
ylabel('F-measure');  
box off;
subplot(1, 4, 3)
plot(1:frame, F_6(1, :), '-', 'color', [0 0.4 0]);
hold on;
plot(1:frame, F_6(6, :), '-', 'color', [150/255  0/255  39/255]);
set(gca, 'XLim', [0, frame], 'YLim', [0, 1.0]);
xlabel('Time (s)'); 
ylabel('F-measure');  
box off;
% average F-measures
subplot(1, 4, 4)
plot(1, F_average(1, 1), 'o', 'MarkerSize', 7, 'color', [0 0.4 0], 'MarkerFaceColor', [0 0.4 0]);
hold on;
plot(2, F_average(1, 2), 'o', 'MarkerSize', 7, 'color', [30/255, 80/255, 173/255], 'MarkerFaceColor', [1, 1, 1], 'MarkerEdgeColor', [30/255, 80/255, 173/255]);
plot(3, F_average(1, 3), 'o', 'MarkerSize', 7, 'color', [30/255, 80/255, 173/255], 'MarkerFaceColor', [30/255, 80/255, 173/255]);
plot(4, F_average(1, 4), 'o', 'MarkerSize', 7, 'color', [220/255, 140/255, 0/255],  'MarkerFaceColor', [1, 1, 1], 'MarkerEdgeColor', [220/255, 140/255, 0/255]);
plot(5, F_average(1, 5), 'o', 'MarkerSize', 7, 'color', [220/255, 140/255, 0/255], 'MarkerFaceColor', [220/255, 140/255, 0/255]);
plot(6, F_average(1, 6), 'o', 'MarkerSize', 7, 'color',  [160/255  0/255  39/255], 'MarkerFaceColor',  [160/255  0/255  39/255]);
plot([1.5, 1.5],[0, 1],'--', 'color', [0.3, 0.3, 0.3], 'linewidth',0.5);
plot([5.5, 5.5],[0, 1],'--', 'color', [0.3, 0.3, 0.3], 'linewidth',0.5);
ylabel('F-measure');  
set(gca, 'XLim', [0, 7],'XTick', 0:1:7, 'XTickLabel', {'', '', '', '', '', '', ''}, 'YLim', [0, 1.0]);
text(1, -0.18, 'EMD', 'rotation',90);%, 'rotation',30
text(2, -0.1, 'Ir', 'rotation',90);
text(3, -0.1, 'Il', 'rotation',90);
text(4, -0.1, 'Id', 'rotation',90);
text(5, -0.1, 'Iu', 'rotation',90);
text(6, -0.13, 'Mo', 'rotation',90);
%===========================================================
