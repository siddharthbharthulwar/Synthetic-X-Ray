function radon()

P = phantom(256);
imshow(P);

theta1 = 0:10:170; 
[R1,~] = radon(P,theta1); 
num_angles_R1 = size(R1,2);

theta2 = 0:5:175;  
[R2,~] = radon(P,theta2);
num_angles_R2 = size(R2,2);



theta3 = 0:2:178;  
[R3,xp] = radon(P,theta3); 
num_angles_R3 = size(R3,2);

P_128 = phantom(128);
[R_128,xp_128] = radon(P_128,theta1);
N_128 = size(R_128,1);

imagesc(theta3,xp,R3);
colormap(hot);
colorbar;
xlabel('Parallel Rotation Angle - \theta (degrees)'); 
ylabel('Parallel Sensor Position - x\prime (pixels)');

end