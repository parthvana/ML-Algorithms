%{
import HW4_Utils
ob=HW4_Utils();
%{ 
[trD, trLb, valD, valLb, trRegs, valRegs]=ob.getPosAndRandomNeg()
[w,b,obj,alpha]=SVM_(trD.',trLb,10);
ob.genRsltFile(w, b, "val", "q3_1")
[ap, prec, rec] = ob.cmpAP("q3_1", "val")
%}

[trD, trLb, valD, valLb, trRegs, valRegs]=ob.getPosAndRandomNeg()
[w,b,obj,alpha]=SVM_(trD.',trLb,10);
trD=trD.'
valD=valD.'
pos=trD(trLb==1,:)
neg=trD(trLb==-1,:)

iteration=10
for i=1:10  
    pred_value=neg.'*w+b;
    neg=neg(pred_value>=-1,:)


pos=[]
neg=[]

[n,d] = size(trD.')
[w,b,obj,alpha]=SVM_(trD.',trLb,10);

for i=1:n
    if trLb(i) == 1
       pos = [pos, trD.'(i,:)];
   else
       neg = [neg, trD.'(i,:)];
   end
end

obj_fn=[];
ap_list=[];

for i=1:10
    
%}
[w, b, obj_list, ap_list] = Hard_mining_v2()    
function [w, b, obj_list, ap_list] = Hard_mining_v2()
    %generating positive and negative training data
    import HW4_Utils
    ob=HW4_Utils();
    [trD, trLb, valD, valLb, trRegs, valRegs] = ob.getPosAndRandomNeg();
    trD=trD.';
    valD=valD.';
    obj_list = [];
    ap_list = [];
    
    pos = trD(trLb>0,:);
    pos_lb = trLb(trLb>0);
    neg = trD(trLb<0,:); 
    neg_lb = trLb(trLb<0);
    
    pos_val = valD(valLb>0,:);
    pos_lb_val = valLb(valLb>0);
    neg_val = valD(valLb<0,:); 
    neg_lb_val = valLb(valLb<0);
    
    pos=vertcat(pos,pos_val);
    pos_lb=vertcat(pos_lb,pos_lb_val);
    neg=vertcat(neg,neg_val);
    neg_lb=vertcat(neg_lb,neg_lb_val);
    
    [w, b, f_0] = SVM_(vertcat(pos,neg), vertcat(pos_lb,neg_lb),10); 
    
    neg=horzcat(neg,-ones(size(neg,1),1));
    
    load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "train"), 'ubAnno');
    ubAnno_train=ubAnno;
    load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, "val"), 'ubAnno');
    ubAnno_val=ubAnno;
    
    
    for iteration = 1:10
        fprintf("Iteration %d",iteration);
     
        temp1 = neg(:,1:1984)*w+b;
        neg(:,1985)=temp1;
        neg=neg(temp1>=-20,:);
      


        rects = cell(1, 93);
        for i=1:93
            im = sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, i);
            im = imread(im);
            [imH, imW,~] = size(im);
            rects{i} = HW4_Utils.detect(im, w, b);
            
            arr = rects{i}(1:5,:);
            pos_arr = cell2mat(ubAnno_train(i));
            
            
            badIdxs = or(arr(3,:) > imW, arr(4,:) > imH);
            arr = arr(:,~badIdxs);
            
           
            for j=1:size(pos_arr,2)
                overlap = ob.rectOverlap(arr, pos_arr(:,j));                    
                arr = arr(:, overlap < 0.18);
                if isempty(arr)
                    break;
                end;
            end;
                
            
            [D_i, R_i] = deal(cell(1, size(arr,2)));
            for j=1:length(D_i)
                ub = arr(:,j);
                ub(1:4)=round(ub(1:4));
                imReg = im(ub(2):ub(4), ub(1):ub(3),:);
                imReg = imresize(imReg, ob.normImSz);
                D_i{j} = ob.cmpFeat(rgb2gray(imReg));
                R_i{j} = imReg;
            end 
            mat = cell2mat(D_i);
            mat = ob.l2Norm(double(mat));
            %disp(size(mat));
            mat=mat.';
            score=arr(5,:);
            score=score.';
            mat=horzcat(mat,score);
            %disp(size(mat));
            %disp(size(neg));
            neg = vertcat(neg, mat);    
        end
        
        rects = cell(1, 92);
        for i=1:92
            im = sprintf('%s/valIms/%04d.jpg', HW4_Utils.dataDir, i);
            im = imread(im);
            [imH, imW,~] = size(im);
            rects{i} = HW4_Utils.detect(im, w, b);
            
            arr = rects{i}(1:5,:);
            pos_arr = cell2mat(ubAnno_val(i));
            
            
            badIdxs = or(arr(3,:) > imW, arr(4,:) > imH);
            arr = arr(:,~badIdxs);
            
           
            for j=1:size(pos_arr,2)
                overlap = ob.rectOverlap(arr, pos_arr(:,j));                    
                arr = arr(:, overlap < 0.18);
                if isempty(arr)
                    break;
                end;
            end;
                
            
            [D_i, R_i] = deal(cell(1, size(arr,2)));
            for j=1:length(D_i)
                ub = arr(:,j);
                ub(1:4)=round(ub(1:4));
                imReg = im(ub(2):ub(4), ub(1):ub(3),:);
                imReg = imresize(imReg, ob.normImSz);
                D_i{j} = ob.cmpFeat(rgb2gray(imReg));
                R_i{j} = imReg;
            end 
            mat = cell2mat(D_i);
            mat = ob.l2Norm(double(mat));
            %disp(size(mat));
            mat=mat.';
            score=arr(5,:);
            score=score.';
            mat=horzcat(mat,score);
            %disp(size(mat));
            %disp(size(neg));
            neg = vertcat(neg, mat);    
        end
        
        
        
        
        [temp,order]=sort(neg(:,1985),'descend');
        neg=neg(order,:);
        max_exp=min(6000,size(neg,1));
        neg=neg(1:max_exp,:);
        disp(size(neg))
        temp_trD = vertcat(pos, neg(:,1:1984));
        temp_ones = -ones(size(neg,1), 1);
        temp_trLb = vertcat(pos_lb, temp_ones);
        [w, b, f_0] = SVM_(temp_trD, temp_trLb,10);
                
        %Calculating Obj values
        obj_list = [obj_list;f_0]
        
        %Calcualting AP values
        ob.genRsltFile(w, b, "val", "temp_val");
        [ap, prec, rec] = ob.cmpAP("temp_val", "val");
        ap_list = [ap_list;ap]
        
        
    end
end
    
    
    
    





    
    

