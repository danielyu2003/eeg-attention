close all;
clear all;

addpath(genpath("C:\Users\yudan\OneDrive\Desktop\eeg_attention\data"));

trial_num=1; % for eye tracking
block_num=1;

% only training data is used?
load(['C:\Users\yudan\OneDrive\Desktop\eeg_attention\data\eeg\Block_' num2str(block_num) '\Training.mat']);

index_EEG=find(diff(double(states.DigitalInput1))~=0);

%double check if the number of EEG index is odd or even after taking out
%the miss trials, if it's odd, it means the trigger value starts at 1 and
%you should delete the first trigger index as well

eeg1=(index_EEG(5)-index_EEG(4));

Eye_all=readtable(['C:\Users\yudan\OneDrive\Desktop\eeg_attention\data\eye\BLOCK_' num2str(block_num) '\TRAINING\Trial_' num2str(trial_num) '.csv']);
index_eye=find(diff(table2array(Eye_all(:,6)))~=0);

%Training
eye1=(index_eye(11)-index_eye(2));

% %Post/Pre
% eye1=(index_eye(6)-index_eye(1));

%sampling frequency for eye
eye_fs=eye1/(eeg1/256);