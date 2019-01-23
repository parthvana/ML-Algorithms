m=10
[w_all,b_all]=multi_SVM(trD.',trLb,10,m)
pred_val=zeros(size(valLb))
pred_label=zeros(size(valLb))
for i=1:m
    pred_temp=valD.'*w_all(:,i)+b_all(i)
    for j=1:length(valLb)
        if pred_temp(j)<pred_val(j)
            pred_label(j)=i
            pred_val(j)=pred_temp(j)
        end 
    end
end

acc_temp=0;
for j=1:length(pred_label)
    if pred_label(j)==valLb(j)
        acc_temp=acc_temp+1
    end
end
acc=acc_temp/length(pred_label)





function [w_all,b_all] = multi_SVM(X,y,C,m)

w_all=[]
b_all=[]
for i=1:m    
    ytemp=y
    ytemp(ytemp==i)=-1
    ytemp(ytemp~=-1)=1
    [w,b,obj,alpha]=SVM_(X,ytemp,10);
    w_all=[w_all,w]
    b_all=[b_all,b]
    
end 
       
end