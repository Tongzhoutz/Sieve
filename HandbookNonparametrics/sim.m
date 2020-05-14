%  This MATLAB file creates Figures 1 and 3 from the paper
%  "Nonparametric Sieve Regression: Least Squares, Averaging Least Squares, and Cross-Validation"
%  by Bruce E. Hansen, University of Wisconsin, www.ssc.wisc.edu/~bhansen
%
%  Simulation setup:
%
%  DGP:       y = g(x) + e          
%          g(x) = a*sin( b*2*pi*x + pi/4 )
%             e ~ N(0,sigma)  
%             x ~ U[0,1]
% 
%  

clear all;
rseed=RandStream('mt19937ar','Seed',12);
RandStream.setGlobalStream(rseed);
options=optimset('algorithm','active-set','Display','off','MaxIter',250);

% parameters
S=10000;                  % number of simulation replications
N=1001;                   % number of grid points on [0,1] 
DGP=2;                    % 1: homoskedastic error 2: heteroskedastic error
b=1;                      % parameter b
X=(0:1/(N-1):1)';         % grid points on [0,1]
R2s=[0.25,0.5,0.75,0.9];  % R-squared  

ns=[50,75,100,125,150,200,300,400,600,800,1000];  % sample size
ks=round(4*(ns.^0.15));                           % splines of order
nk=length(ns);
               
for r=1:4;
    R2=R2s(r);                % R-squared
    a=sqrt(2*R2/(1-R2));      % parameter a
    G=a*sin(b*2*pi*X+pi/4);   % g(x)
    imse=zeros(nk,ks(end)+6);  
    
for k=1:nk;
    n=ns(k);    
    K=ks(k);
    M=K+1;
    t=(0:(1/M):1);
    t1=t(2);
    t2=t(end-1);
    t3=t(end-2);
    mseA=zeros(S,M+5);
    
    for s=1:S;
        if floor(s/100)*100==s; disp([n,s]); end
        % DGP
        x=rand(n,1);
        while sum((x-t1)<-0.01)<2||sum((x-t2)>0.01)<2||sum((x-t3)>0.01)<3;
            x=rand(n,1); 
        end
        e=randn(n,1);
        if DGP==2; e=sqrt(5*x.^4).*e; end;
        g=a*sin(b*2*pi*x+pi/4);
        y=g+e;
    
        % OLS Estimation 
        x1=[ones(n,1),x,x.^2];
        X1=[ones(N,1),X,X.^2];
        aic=zeros(M,1);
        aicc=zeros(M,1);
        ee=zeros(n,M);
        GG=zeros(N,M);
        for i=1:M;
            ti=(0:(1/i):1);
            ti=ti(2:end-1);
            x2=(x*ones(1,i-1)-ones(n,1)*ti);
            x2=(x2.^2).*(x2>0);
            xi=[x1,x2];            
            if rank(xi'*xi)<size(xi,2);
                beta=pinv(xi'*xi)*xi'*y;
            else
                beta=(xi'*xi)\xi'*y;
            end 
            ei=y-xi*beta;            
            p=length(beta);
            sig2=(ei'*ei)/n;
            aic(i)=n*log(sig2)+2*p;
            aicc(i)=n*log(sig2)+n*(n+p)/(n-p-2);
            if rank(xi'*xi)<size(xi,2);
                hi=diag(xi*pinv(xi'*xi)*xi');
            else
                hi=diag(xi/(xi'*xi)*xi');
            end 
            ee(:,i)=ei.*(1./(1-hi));
            X2=(X*ones(1,i-1)-ones(N,1)*ti);
            X2=(X2.^2).*(X2>0);
            Xi=[X1,X2];
            GG(:,i)=Xi*beta;
        end

        % infeasible optimal weights
        w0=ones(M,1)/M;
        a1=GG'*GG;
        a2=-GG'*G;
        w_opt=quadprog(a1,a2,zeros(1,M),0,ones(1,M),1,zeros(M,1),ones(M,1),w0,options);
        w_opt=w_opt.*(w_opt>0);
        w_opt=w_opt/sum(w_opt);
        G0=GG*w_opt;
        
        % JMA
        a1=ee'*ee;
        a2=zeros(M,1);
        w_jma=quadprog(a1,a2,zeros(1,M),0,ones(1,M),1,zeros(M,1),ones(M,1),w0,options);
        w_jma=w_jma.*(w_jma>0);
        w_jma=w_jma/sum(w_jma);
        GJ=GG*w_jma;

        % AIC, AICc, CV
        w_aic=0+logical(aic==min(aic));
        w_aicc=0+logical(aicc==min(aicc)); 
        cv=diag(a1)/n;
        w_cv=0+logical(cv==min(cv));
    
        % MSE
        mseM=mean((GG-G*ones(1,M)).^2);  % Splines
        mse0=mean((G0-G).^2);            % Optimal
        mse1=mseM(w_aic>0);              % AIC
        mse2=mseM(w_aicc>0);             % AICc
        mse3=mseM(w_cv>0);               % CV   
        mse4=mean((GJ-G).^2);            % JMA
        mseA(s,:)=[mse0,mse1,mse2,mse3,mse4,mseM];
    end
imse(k,1:M+5)=mean(mseA);
end

% figure
imse_opt=imse(:,1);
nn=log(ns);

figure(1);
subplot(2,2,r);
hold on;
box on;
title(['R^{2} = ',num2str(R2)],'FontSize',14);
plot(nn,imse(:,7)./imse_opt,'-r*','MarkerSize',5);
plot(nn,imse(:,8)./imse_opt,'LineStyle','-.','color',[0,0.6,0],'LineWidth',2);
plot(nn,imse(:,9)./imse_opt,'k','LineWidth',2);
plot(nn,imse(:,10)./imse_opt,'--r','LineWidth',2);
plot(nn,imse(:,11)./imse_opt,'LineStyle',':','color',[0,0,0.8],'LineWidth',2);
set(gca,'XTick',nn);
set(gca,'XTickLabel',ns);
if r==4; legend('Spline(1)','Spline(2)','Spline(3)','Spline(4)','Spline(5)','Location','NorthEast'); end
axis([min(nn) max(nn) 1 3]);
xlabel('n','FontSize',12);
ylabel('IMSE','FontSize',12);
hold off;

figure(2);
subplot(2,2,r); 
hold on;
box on;
title(['R^{2} = ',num2str(R2)],'FontSize',14);
plot(nn,imse(:,2)./imse_opt,'LineStyle',':','color',[0,0,0.8],'LineWidth',2.2);
plot(nn,imse(:,3)./imse_opt,'LineStyle','-.','color',[0,0.6,0],'LineWidth',2);
plot(nn,imse(:,4)./imse_opt,'--r','LineWidth',2);
plot(nn,imse(:,5)./imse_opt,'k','LineWidth',2);
set(gca,'XTick',nn);
set(gca,'XTickLabel',ns);
if r==4; legend('AIC','AICc','CV','JMA','Location','NorthEast'); end; 
axis([nn(1) nn(11) 1 4]);
xlabel('n','FontSize',12);
ylabel('IMSE','FontSize',12);
hold off;

end