% % load in the predicted labels and their index.
clc
clear
mullabel=load('/home/lein/caffe/myfinetunning/data/mulpre_labels.mat');
mullabel=mullabel.data';
mul_index=load('/home/lein/caffe/myfinetunning/data/mul_index.mat');
mul_index=mul_index.data';
mul(:,1)=mul_index;
mul(:,2)=mullabel;
%index and labels
firlabel=load('/home/lein/caffe/myfinetunning/data/firpre_labels.mat');
firlabel=firlabel.data';
fir_index=load('/home/lein/caffe/myfinetunning/data/fir_index.mat');
fir_index=fir_index.data';
fir(:,1)=fir_index;
fir(:,2)=firlabel;

im=imread('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/RH.tif');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I332=imread('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/MR30_RHRVRRRL.tif');
I332=im2uint8(I332)*10;
RH=I332(:,:,1);
RV=I332(:,:,2);
RL=I332(:,:,3);

label_332= load('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_332/label_332.mat');
label=cell2mat(struct2cell(label_332));


count_1=0;
count_m=0;


firstice=[];
mulice=[];
for i=1:size(label)
    i1=label(i,1);
    j1=label(i,2);
    if label(i,3)==6
        % first year ice 
        
        count_1=count_1+1; 
        firstice(count_1,1)= i1;
        firstice(count_1,2)= j1;
     
        %fname=strcat('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_444/first_year_patch/','first_year_ice_scene444_',strcat(num2str(count_1),'.jpg'));
        %imwrite(D,fname);
    end;
       if label(i,3)==12;
         %multiyear ice
       
        count_m=count_m+1; 
        mulice(count_m,1)=i1;
        mulice(count_m,2)=j1;
    
        
       % fname=strcat('/home/lein/Documents/remote_sensing/project/ice_cp_data/CP_9x9_444/multi_year_patch/','multi_year_ice_scene444_',strcat(num2str(count_m),'.jpg'));
        %imwrite(D,fname);
         
    
     end;
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%index with label defined as fir_final
fir_final=[];
mul_final=[];
for iii=1:size(firlabel);
    
    fir_final(iii,:)=firstice(fir(iii,1),:);
end;
fir_final(:,3)=firlabel;

for jjj=1:size(mullabel)
    mul_final(jjj,:)=mulice(mul(jjj,1),:);
    
end;
mul_final(:,3)=mullabel;
total_final=[fir_final;mul_final];
kk=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot 
for ij=1:size(total_final)
    if total_final(ij,3)==0
        fi_index(kk,:)=total_final(ij,:);
        kk=kk+1;
    end;
end;
vv=1;
for ij=1:size(total_final)
    if total_final(ij,3)==1
        mu_index(vv,:)=total_final(ij,:);
        vv=vv+1;
    end;
end;


figure; imshow(im)


figure; imshow(im)
hold on;

 plot(fi_index(:,2),fi_index(:,1),'ro')
  plot(mu_index(:,2),mu_index(:,1),'bo')


