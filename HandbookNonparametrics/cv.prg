/* CV.PRG */

/* 
This Gauss program creates Figures 2 and 4, and Table 1, from
"Nonparametric Sieve Regression: Least Squares, Averaging Least Squares, and Cross-Validation"
by Bruce E. Hansen
University of Wisconsin
www.ssc.wisc.edu/~bhansen
*/

library pgraph;

graphset;
pqgwin many;
let _pmcolor = 0 0 0 0 0 0 0 0 15;
_pcolor=1;
fonts("complex simgrma");
_pdate="";
_plwidth=6;

n=200;
r2=.5;
k=6;

a=sqrt(2*r2/(1-r2));
xn=101;
xs=seqa(0,.01,xn);
g=a*sin(xs*2*pi+pi/4);


load x;
load y;

/*
This loads the "data" x and y from x.fmt and y.fmt used in the actual figures
If you want to replicate an analog using random data, comment out the above two lines and replace with the following two lines

x=rndu(n,1);
y=a*sin(x*2*pi+pi/4)+rndn(n,1).*sqrt((x.^2)*sqrt(5));

*/


cv=zeros(k,1);
es=zeros(n,k);
gg=zeros(xn,k);

z0=ones(n,1)~x~(x.^2);
z0s=ones(xn,1)~xs~(xs.^2);

for ki (1,k,1);

if ki>1;
 kn=seqa(1/ki,1/ki,ki-1)';
 z=z0~(((x-kn).^2).*(x.>kn));
 zs=z0s~(((xs-kn).^2).*(xs.>kn));
else;
 z=z0;
 zs=z0s;
endif;
zz=invpd(z'z);
beta=zz*(z'y);
ehat=y-z*beta;
h=(z*zz).*z;
h=sumc(h');
et=ehat./(1-h);
es[.,ki]=et;
cv[ki,1]=(et'et)/n;
gg[.,ki]=zs*beta;

endfor;


/* Figure 2 in paper */
kk=seqa(0,1,k);
xtics(0,k-1,1,1);
xlabel("Number of Knots");
ylabel("Cross-Validation Function");
xy(kk,cv);

ki=minindc(cv);
kk~cv;
"Minimizing order " ki; 
"";

ss=(es'es)/n;
w0=ones(k,1)/k;
{w,u1,u2,u3,u4,ret}=QProg(w0,ss,zeros(k,1),ones(1,k),1,0,0,0~1);
w=w.*(w.>(1e-8));
w=w/sumc(w);
w; 

/* Table 1 from paper */
ghat=gg[.,ki];
gw=gg*w;

/* Figure 4 in paper */

let _pltype = 6 3 1;
let _plegctl= 1 5 .6 .93;
_plegstr="g(x)\000CV Estimate\000JMA Estimate";
xtics(0,1,.1,10);
xlabel("x");
ylabel("g(x)");
xy(xs,g~ghat~gw);
