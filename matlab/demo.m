disp('======= KITTI 2015 Benchmark Demo =======');
clear all; close all; dbstop error;
% error threshold
tau = [3 0.05];

d_err_t = 0;
d_err_tQ = 0;
d_err_tD = 0;

for x = 1:40
    imageL = im2double(imread(strcat('predictions/',num2str(1),'/imgL.png')));
    gt = im2double(imread(strcat('predictions/',num2str(1),'/dispgt.png')));
    
    % stereo demo
    disp('Load and show disparity map ... ');
    D_est = disp_read(['predictions/' num2str(x) '/prediction.png']);
    est = im2double(imread(strcat('predictions/',num2str(1),'/prediction.png')));
    D_estQ = disp_read(['predictions/' num2str(x) '/predictionQ.png']);
    estQ = im2double(imread(strcat('predictions/',num2str(1),'/predictionQ.png')));
    D_estD = disp_read(['predictions/' num2str(x) '/predictionD.png']);
    estD = im2double(imread(strcat('predictions/',num2str(1),'/predictionD.png')));

    D_gt  = disp_read(['predictions/' num2str(x) '/dispgt.png']);
    
    d_err = disp_error(D_gt,D_est,tau);
    d_errQ = disp_error(D_gt,D_estQ,tau);
    d_errD = disp_error(D_gt,D_estD,tau);

    d_err_t = d_err_t + d_err;
    d_err_tQ = d_err_tQ + d_errQ;
    d_err_tD = d_err_tD + d_errD;
    
    if x==5
        D_err = disp_error_image(D_gt,D_est,tau);
        D_errQ = disp_error_image(D_gt,D_estQ,tau);
        D_errD = disp_error_image(D_gt,D_estD,tau);

        out = [est(:,:,[1 1 1]);disp_to_color([D_est;D_gt]);D_err];
        outQ = [estQ(:,:,[1 1 1]);disp_to_color([D_estQ;D_gt]);D_errQ];
        outD = [estD(:,:,[1 1 1]);disp_to_color([D_estD;D_gt]);D_errD];
       
        subplot(2,3,2),imshow([imageL; gt(:,:,[1 1 1])]);
        title('Reference');
        subplot(2,3,4),imshow([out]);
        title(sprintf('DispNet Mejorada - Pytorch - Disparity Error: %.2f %%',d_err*100));
        subplot(2,3,5),imshow([outD]);
        title(sprintf('DispNet - Pytorch - Disparity Error: %.2f %%',d_errD*100));
        subplot(2,3,6),imshow([outQ]);
        title(sprintf('DispNet Mejorada - Alveo U50 - Disparity Error: %.2f %%',d_errQ*100));
    end
end

d_err_t = 100*(d_err_t/40)
d_err_tQ = 100*(d_err_tQ/40)
d_err_tD = 100*(d_err_tD/40)