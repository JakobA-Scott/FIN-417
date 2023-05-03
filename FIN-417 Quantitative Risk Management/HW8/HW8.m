clear all
close all

%Read file
table = readtable('Global_3_Factors_Daily.csv','PreserveVariableNames',true);
data1=table2array(table(:,2));
data2=table2array(table(:,3));
len=length(data1);
[sorted_vals1,indices1] = sort(data1,'ascend');
[sorted_vals2,indices2] = sort(data2,'ascend');
r11 = 1:len;
r1(indices1) = r11;
F1=r1/(len+1);
r22 = 1:len;
r2(indices2) = r22;
F2=r2/(len+1);
U=[F1',F2'];
% scatter(F1,F2)
% xlabel('SMB')
% ylabel('HML')

initial_theta=6;
fun=@(theta) gumbel_target_function(U,theta);
[theta,negative_function_value]=fmincon(fun,initial_theta,[],[],[],[],1,Inf) % Change lower bound depending on the family.
function target=target_function(U,theta)
gumbelpdf=copulapdf('Gumbel',U,theta); % Change here based on which copula family is being optimized.
boolean=gumbelpdf>0;                     % To avoid problems in log function.
gumbelpdf=nonzeros(gumbelpdf.*boolean);  % To avoid problems in log function.
target=-sum(log(gumbelpdf));           % Maximizing log-lilelihood is the same as minimizing negative log-likelihood.
end