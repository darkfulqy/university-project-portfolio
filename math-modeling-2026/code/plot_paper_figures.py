from pathlib import Path
import json, re, hashlib, csv, html, zipfile, math
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, colors
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE / 'figures/recreated'
INPUT = PACKAGE / 'data'
for folder in ['PDF','数据与代码']:
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
FONT = Path('/System/Library/Fonts/Supplemental/Songti.ttc')
if FONT.exists():
    fm.fontManager.addfont(str(FONT))
    CJK = fm.FontProperties(fname=str(FONT)).get_name()
else:
    CJK = next((n for n in ['Noto Serif CJK SC', 'Source Han Serif SC', 'SimSun']
                if any(f.name == n for f in fm.fontManager.ttflist)), 'serif')

BLUE, TEAL, ORANGE, PURPLE = '#0072B2', '#009E73', '#D55E00', '#8B5FA2'
INK, GRAY, GRID = '#20252B', '#65717B', '#E2E6EA'
TIME_COLORS = [BLUE, TEAL, ORANGE]
MARKERS = ['o', 's', '^']
STYLES = ['-', '--', '-.']
plt.rcParams.update({
    'font.family': [CJK, 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.edgecolor': INK, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': .7, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.5, 'lines.markersize': 4,
    'legend.frameon': False, 'legend.handlelength': 2.5,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
    'axes.unicode_minus': False,
})
RECORDS = []


def axes_clean(ax, x=None, y=None, xlim=None, ylim=None, grid=True):
    if x: ax.set_xlabel(x)
    if y: ax.set_ylabel(y)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    if grid: ax.grid(axis='y', color=GRID, lw=.55, ls=(0, (2, 3)))
    ax.tick_params(direction='out', pad=3)

def sublabel(ax, text):
    ax.set_title(text, loc='left', pad=9)

def diagram(size=(6.5, 2.8), xlim=(0, 10), ylim=(0, 4.3)):
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis('off')
    return fig, ax

def txt(ax, x, y, s, c=INK, fs=10, ha='center', **kw):
    return ax.text(x, y, s, ha=ha, va='center', fontsize=fs, color=c, **kw)

def arrow(ax, a, b, c=INK, lw=1.15, style='-|>', scale=10, **kw):
    p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=scale,
                        linewidth=lw, color=c, shrinkA=0, shrinkB=0, **kw)
    ax.add_patch(p)
    return p

def box(ax, xy, w, h, color=BLUE, fill='#F4F8FA'):
    p=Rectangle(xy,w,h,facecolor=fill,edgecolor=color,lw=.9)
    ax.add_patch(p)
    return p


TABLES=json.loads((INPUT/'figure_tables.json').read_text())
SOURCE_HASH=hashlib.sha256((INPUT/'figure_tables.json').read_bytes()).hexdigest()

def number(s):
    m=re.search(r'[+-]?\d+(?:\.\d+)?',s.replace('−','-').replace('**',''))
    assert m,s
    return float(m[0])

def numeric_table(j):
    return np.array([[number(v) for v in row] for row in TABLES[j]['rows'][1:]],float)

Q1T,Q1C,Q2T,Q2C,Q3,Q4,BASE = (numeric_table(j) for j in [4,5,7,8,9,10,11])
assert Q2T.shape==(6,6) and Q2C.shape==(6,6) and Q4.shape==(9,6)
RADII=np.array([0,.5,1,1.5,2.])

ENV=np.load(INPUT/'processed/attachment1.npy')
OBS=np.load(INPUT/'processed/attachment2.npy')

def save(fig,code,title,caption,source,group,payload=None):
    for ax in fig.axes:
        for label in ax.get_xticklabels()+ax.get_yticklabels():
            if not re.search(r'[\u4e00-\u9fff]',label.get_text()):
                label.set_fontfamily('Times New Roman')
    fig.tight_layout(pad=.9)
    stem=f'{code}_{title}'
    for fmt in ['PDF']:
        fig.savefig(ROOT/fmt/(stem+'.'+fmt.lower()),
                    dpi=600 if fmt=='PNG' else 300,bbox_inches='tight',pad_inches=.06,
                    metadata={'Creator':'Final manuscript figures; Matplotlib'} if fmt=='PDF' else None)
    traces=[]
    for ax in fig.axes:
        for line in ax.lines:
            try:
                x=np.asarray(line.get_xdata(),float);y=np.asarray(line.get_ydata(),float)
                traces.append({'label':line.get_label(),'x':x.tolist(),'y':y.tolist()})
            except (ValueError,TypeError):pass
    image_arrays=[np.asarray(im.get_array()).tolist() for ax in fig.axes for im in ax.images]
    bar_widths=[float(p.get_width()) for ax in fig.axes for p in ax.patches] if code=='10-01' else []
    texts=[t.get_text() for ax in fig.axes for t in ax.texts]
    RECORDS.append({'code':code,'title':title,'stem':stem,'caption':caption,
                    'source':source,'group':group,'payload':payload,'traces':traces,
                    'image_arrays':image_arrays,'bar_widths':bar_widths,'texts':texts})
    plt.close(fig)
    print('完成',stem,flush=True)

def attachment_figures():
    for code,col,title,label,color in [
        ('A01',1,'烘房温度',r'烘房温度 $T_a$ / °C',ORANGE),
        ('A02',2,'烘房水分浓度',r'烘房水分浓度 $C_a$ / (kg/kg)',BLUE)]:
        fig,axs=plt.subplots(1,2,figsize=(6.5,2.5),gridspec_kw={'width_ratios':[1.45,1]})
        for ax,mask,unit,xlabel,panel in [
            (axs[0],np.ones(len(ENV),bool),3600,'时间 / h','(a) 0–4 h'),
            (axs[1],ENV[:,0]<=1800,60,'时间 / min','(b) 前30分钟')]:
            ax.plot(ENV[mask,0]/unit,ENV[mask,col],c=color,marker='o',ms=2.3,
                    markevery=8 if unit==3600 else 1,mfc='white',mew=.7,lw=1.2)
            axes_clean(ax,xlabel,label);sublabel(ax,panel)
        axs[0].set_xticks([0,1,2,3,4]);axs[1].set_xticks([0,10,20,30])
        save(fig,code,title,'附件1中的原始观测记录及前30分钟放大图。线段连接相邻记录，完整时段的圆点稀疏显示。',
             '附件1；正文第7.2节','附件观测',{'attachment':1,'column':col})
    radius_input('A03',12,'药材实测半径','附件观测')
    rate=100*(OBS[0,1]-OBS[:,1])/OBS[0,1]
    fig,ax=plt.subplots(figsize=(6.5,2.65))
    ax.plot(OBS[:,0]/3600,rate,c=TEAL,lw=1.7)
    for h in [6,24,72]:
        j=np.flatnonzero(OBS[:,0]==h*3600)[0]
        ax.plot(h,rate[j],'o',c=TEAL,mfc='white')
        ax.annotate(f'{rate[j]:.1f}%',(h,rate[j]),xytext=(-5,-17),textcoords='offset points',
                    ha='right',fontsize=9,c=TEAL)
    axes_clean(ax,'时间 / h','整体径向收缩率 / %',(0,75),(0,44));ax.set_xticks([0,12,24,36,48,60,72])
    save(fig,'A04','整体径向收缩率',r'由附件2按 $[R_0-R(t)]/R_0\times100\%$ 计算整体径向收缩率；该量描述半径变化。',
         '附件2；正文第9.4节','附件观测',{'attachment':2,'formula':'100*(R0-R)/R0'})

def radius_input(code,zoom,title,group):
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.5),gridspec_kw={'width_ratios':[1.45,1]})
    for ax,end,panel in [(axs[0],72,'(a) 完整观测区间'),(axs[1],zoom,f'(b) 前{zoom}小时')]:
        m=OBS[:,0]<=end*3600
        ax.plot(OBS[m,0]/3600,OBS[m,1],c=TEAL,marker='o',ms=2.8,mfc='white',mew=.8,
                markevery=4 if end==72 else 1)
        axes_clean(ax,'时间 / h',r'外半径 $R$ / cm',(-.02*end,1.02*end));sublabel(ax,panel)
        ax.set_xticks([0,12,24,36,48,60,72] if end==72 else [0,zoom/3,2*zoom/3,zoom])
    save(fig,code,title,'附件2的半径观测与相邻观测之间的分段线性连接。实测半径作为均匀收缩模型的几何输入。',
         '附件2；正文第9.4、9.5节',group,{'attachment':2})

def cylinder():
    fig,ax=diagram((6.5,2.75),ylim=(0,4.2));ax.set_aspect('equal')
    x1,x2,cy,ry,rx=.95,3.8,2.05,.85,.33
    ax.add_patch(Rectangle((x1,cy-ry),x2-x1,2*ry,fc='#F4F8FA',ec='none'))
    for x,fill in [(x1,'white'),(x2,'#EAF3F8')]:ax.add_patch(Ellipse((x,cy),2*rx,2*ry,fc=fill,ec=BLUE,lw=1.1))
    for y in [cy-ry,cy+ry]:ax.plot([x1,x2],[y,y],c=BLUE,lw=1.1)
    ax.plot([.45,4.35],[cy,cy],c=GRAY,lw=.75,ls='--')
    arrow(ax,(x1,.72),(x2,.72),GRAY,style='<->');txt(ax,2.35,.41,r'$L=25\ \mathrm{cm}$',fs=10)
    txt(ax,2.35,3.97,'(a) 圆柱药材')
    cx,cy,R=7.35,2.05,1.12
    for rad,fill,edge in [(R,'#F7FAFB',BLUE),(.7,'#DDEFEA',TEAL),(.59,'#F7FAFB',TEAL)]:
        ax.add_patch(Circle((cx,cy),rad,fc=fill,ec=edge,lw=1))
    ax.plot(cx,cy,'o',c=INK,ms=2.5);arrow(ax,(cx,cy),(cx+R,cy),BLUE)
    txt(ax,cx+.6,cy-.22,r'$R_0=2\ \mathrm{cm}$',BLUE,9)
    arrow(ax,(cx,cy),(cx-.45,cy+.39),TEAL);txt(ax,cx-.17,cy+.33,r'$r$',TEAL)
    ax.annotate(r'环元厚度 $\mathrm{d}r$',(cx-.46,cy+.48),(5.15,2.97),fontsize=9,color=TEAL,
                arrowprops={'arrowstyle':'-','color':TEAL,'lw':.8})
    arrow(ax,(cx,3.54),(cx,3.02),ORANGE);txt(ax,8.43,3.35,'热量传入',ORANGE,9)
    arrow(ax,(cx,.93),(cx,.33),BLUE);txt(ax,8.46,.65,'水分排出',BLUE,9)
    txt(ax,7.35,3.97,'(b) 径向环元与表面交换')
    txt(ax,5,.06,r'单位长度环元体积：$\mathrm{d}V/L=2\pi r\,\mathrm{d}r$',fs=9)
    save(fig,'S01','圆柱传热传质示意图','药材长度25 cm、初始半径2 cm。图示径向传热传质近似及单位长度环元，几何示意不按真实长径比绘制。',
         '正文第1.2、3、6.1节','物理模型')

def coupling():
    fig,ax=diagram((6.5,2.65),ylim=(0,4.1))
    for x,col,title,detail in [(.5,ORANGE,r'温度场 $T(r,t)$','内部导热 · 表面对流'),(6.8,BLUE,r'含水率场 $C(r,t)$','内部扩散 · 表面交换')]:
        box(ax,(x,1.35),2.7,1.15,col,'white');txt(ax,x+1.35,2.13,title,col,11);txt(ax,x+1.35,1.65,detail,fs=9)
    arrow(ax,(3.2,2.22),(6.8,2.22),ORANGE);txt(ax,5,2.53,r'$T\ \rightarrow\ D(C,T)\ \rightarrow$ 水分迁移',ORANGE,9)
    arrow(ax,(6.8,1.53),(3.2,1.53),BLUE);txt(ax,5,1.17,r'$C\ \rightarrow\ \rho(C)c_p(C),k(C)$',BLUE,9)
    for x,col,label in [(1.85,ORANGE,r'环境温度 $T_a(t)$'),(8.15,BLUE,r'环境水分边界 $C_a(t)$')]:
        arrow(ax,(x,3.3),(x,2.5),col);txt(ax,x,3.6,label,col,9)
    txt(ax,5,.36,r'扩散系数中的 $T$ 使用 K；摄氏温度须先加 273.15',fs=9)
    save(fig,'S02','水热耦合机制图','温度通过扩散系数影响水分迁移，含水率通过体积热容与导热系数影响温度演化。扩散系数使用绝对温度，数值结果图以摄氏度显示。',
         '正文符号说明、第7.1节及第9.4节','物理模型')

def shrink_map():
    fig,ax=diagram((6.5,2.75),ylim=(0,4.15));ax.set_aspect('equal')
    for x,R,name in [(2.2,1.17,'(a) 初始截面'),(7.65,.702,'(b) 收缩后截面')]:
        for frac in [1,.75,.5,.25]:ax.add_patch(Circle((x,2.2),R*frac,fc='none',ec=BLUE if frac==1 else TEAL,lw=1,ls='-' if frac==1 else '--'))
        px,py=x+.75*R*np.cos(np.pi/5),2.2+.75*R*np.sin(np.pi/5)
        ax.plot(x,2.2,'o',c=INK,ms=2.5);ax.plot(px,py,'o',c=ORANGE,ms=5);arrow(ax,(x,2.2),(px,py),ORANGE)
        txt(ax,x,3.88,name);txt(ax,x,.42,r'$r_0=R_0\xi$' if x<5 else r'$r(t)=R(t)\xi$',fs=11)
        txt(ax,x,.79,r'$R_0=2.0\ \mathrm{cm}$' if x<5 else r'示例：$R(t)=1.2\ \mathrm{cm}$',fs=9)
    arrow(ax,(3.92,2.2),(6.18,2.2),GRAY);txt(ax,5.05,2.65,'均匀径向收缩');txt(ax,5.05,1.72,r'$\lambda(t)=R(t)/R_0$',fs=11)
    txt(ax,5,.01,r'同一材料点：$\xi=r/R(t)$ 固定，$r(t)/r_0=R(t)/R_0$',fs=9)
    save(fig,'S03','均匀收缩坐标映射图','均匀径向收缩假设下的材料点映射。外半径从2.0 cm到1.2 cm仅作为收缩示例，同色点具有相同随体标签。',
         '正文第9.4节，式(9-21)至(9-23)','物理模型')

def grid_2d():
    fig,ax=diagram((6.5,3.1),ylim=(0,4.9));x0,x1,y0,y1=2.9,7.35,1,3.76;dx=(x1-x0)/6;dz=(y1-y0)/6
    for j in range(7):
        ax.plot([x0+j*dx]*2,[y0,y1],c=GRID,lw=.7);ax.plot([x0,x1],[y0+j*dz]*2,c=GRID,lw=.7)
    ax.plot([x0,x0],[y0,y1],c=BLUE,lw=1.6);ax.plot([x1,x1],[y0,y1],c=ORANGE,lw=1.6)
    for y in [y0,y1]:ax.plot([x0,x1],[y,y],c=TEAL,lw=1.5)
    for i in range(7):
        for j in range(7):ax.plot(x0+i*dx,y0+j*dz,'o',c=GRAY,ms=2)
    cx,cy=x0+3*dx,y0+3*dz
    for x,y in [(cx-dx,cy),(cx+dx,cy),(cx,cy-dz),(cx,cy+dz)]:ax.plot([cx,x],[cy,y],c=BLUE,lw=1.5);ax.plot(x,y,'o',c=BLUE,ms=4)
    ax.plot(cx,cy,'o',c=ORANGE,ms=5);txt(ax,cx+.18,cy+.22,r'$(i,j)$',fs=9)
    arrow(ax,(x0-.26,y0-.25),(x1+.42,y0-.25),GRAY);arrow(ax,(x0-.26,y0-.25),(x0-.26,y1+.35),GRAY)
    txt(ax,x1+.55,y0-.25,r'$r$',fs=11);txt(ax,x0-.26,y1+.48,r'$z$',fs=11)
    txt(ax,1.2,2.57,r'中心轴 $r=0$',BLUE,9);txt(ax,1.2,2.16,r'$\partial u/\partial r=0$',BLUE,10)
    txt(ax,8.7,2.57,r'侧面 $r=R$',ORANGE,9);txt(ax,8.7,2.16,'对流交换',ORANGE,9)
    txt(ax,5.2,4.25,r'$z=L$：端面交换',TEAL,9);txt(ax,5.2,.39,r'$z=0$：端面交换',TEAL,9)
    txt(ax,5,.02,r'轴对称模型；$u$ 表示温度或含水率',GRAY,9)
    save(fig,'5-01','二维轴对称有限差分网格','用于预热阶段端部效应验证的二维轴对称网格，中心轴对称，侧面及两端面采用对流边界。图中网格密度仅作示意。',
         '正文第5.2.1、5.2.2节','一维假设验证')

def explicit_stencil():
    fig,ax=diagram((6.5,2.95),ylim=(0,4.5));cx,cy,d=2.8,2.6,.78
    for x,y,label in [(cx-d,cy,r'$(i-1,j)$'),(cx+d,cy,r'$(i+1,j)$'),(cx,cy+d,r'$(i,j+1)$'),(cx,cy-d,r'$(i,j-1)$')]:
        ax.plot([cx,x],[cy,y],c=BLUE,lw=1.2);ax.plot(x,y,'o',c=BLUE,ms=4)
        txt(ax,x,y+(.24 if y>=cy else -.25),label,fs=9)
    ax.plot(cx,cy,'o',c=BLUE,ms=5);txt(ax,cx+.28,cy-.29,r'$u^n_{i,j}$',fs=10)
    txt(ax,cx,4.04,r'已知时间层 $t_n$',BLUE,10)
    arrow(ax,(4.2,cy),(6.45,cy),GRAY);txt(ax,5.3,3.08,r'$\Delta t=0.05\ \mathrm{s}$',fs=10)
    ax.plot(7.45,cy,'o',c=ORANGE,ms=6);txt(ax,7.45,3.03,r'$u^{n+1}_{i,j}$',ORANGE,12)
    txt(ax,7.45,4.04,'显式更新中心节点',ORANGE,10)
    txt(ax,5,.65,r'$\dfrac{u^{n+1}_{i,j}-u^n_{i,j}}{\Delta t}=\mathcal{L}_h[u^n]_{i,j}$',fs=13)
    txt(ax,5,.03,'柱坐标径向算子与轴心镜像条件按正文单独处理',GRAY,9)
    save(fig,'5-02','二维显式差分推进模板','温度与含水率均采用时间前向欧拉和空间中心差分，时间步长为0.05 s。示意已知时间层的五点信息用于更新下一时间层的中心节点。',
         '正文第5.2.2节，式(5-13)至(5-15)','一维假设验证')

def end_effects():
    rowsT=TABLES[1]['rows'][1:];rowsC=TABLES[2]['rows'][1:]
    dist=np.array([number(r[1]) for r in rowsT]);T=np.array([number(r[2]) for r in rowsT]);C=np.array([number(r[2]) for r in rowsC]);order=np.argsort(dist)
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.9))
    for ax,values,col,label,panel,dec in [(axs[0],T,ORANGE,'中心温度 / °C','(a) 温度端部效应',2),(axs[1],C,BLUE,'中心含水率 / (kg/kg)','(b) 水分端部效应',4)]:
        ax.plot(dist[order],values[order],c=col,marker='o',mfc='white',mew=1)
        axes_clean(ax,'到端面距离 / cm',label,(-.5,13.2));sublabel(ax,panel)
        for j in range(len(dist)):
            offset=(-7,-14) if j==2 and dec==2 else (5,8 if j!=2 else -15)
            ax.annotate(f'{values[j]:.{dec}f}',(dist[j],values[j]),xytext=offset,textcoords='offset points',fontsize=8.3,ha='right' if j==0 or (j==2 and dec==2) else 'left')
        ax.margins(y=.23)
    save(fig,'5-03','预热阶段端部效应对比','在1800 s时刻，比较正文列出的四个轴向截面中心温度与中心含水率。圆点为表值，线段仅连接表列位置。',
         '正文第5.2.3节两张截面对照表','一维假设验证',{'tables':[1,2]})

def radial_plot(code,title,data,indices,unit,source,group,table_id):
    fig,ax=plt.subplots(figsize=(6.5,2.8))
    for j,i in enumerate(indices):
        t=data[i,0];lab=f'{t:g} '+('s' if group=='预热阶段' else 'h')
        ax.plot(RADII,data[i,1:],c=TIME_COLORS[j],ls=STYLES[j],marker=MARKERS[j],mfc='white',mew=1,label=lab)
    axes_clean(ax,r'距中心距离 $r$ / cm',unit,(-.03,2.03));ax.set_xticks(RADII)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,borderaxespad=0,columnspacing=2.5)
    ax.margins(y=.12)
    save(fig,code,title,'各时刻径向分布直接取自终稿对应表格。每条曲线仅含表列的五个径向位置，线段用于连接数据点。',source,group,{'table':table_id,'row_indices':indices,'radii_cm':RADII.tolist()})

def properties():
    c=np.linspace(0,2.55,401);d=np.zeros_like(c);d[1:]=2.4e-3*np.exp(-.45/c[1:]-3850/323.15)*1e9
    vals=[650+128*c,1450+2736*c/(1+c),.21+.38*c/(1+c),d]
    labels=[r'密度 $\rho$ / (kg/m$^3$)',r'比热容 $c_p$ / (J/(kg·K))',r'导热系数 $k$ / (W/(m·K))',r'扩散系数 $D$ / ($10^{-9}$ m$^2$/s)']
    forms=[r'$\rho=650+128C$',r'$c_p=1450+2736C/(1+C)$',r'$k=0.21+0.38C/(1+C)$',r'$T=323.15\ \mathrm{K}$（50 °C）']
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.05))
    for j,(ax,v,label,formula,col) in enumerate(zip(axs.flat,vals,labels,forms,[BLUE,ORANGE,TEAL,PURPLE])):
        ax.plot(c,v,c=col,lw=1.7);axes_clean(ax,r'干基含水率 $C$ / (kg/kg)',label,(0,2.6));sublabel(ax,f'({chr(97+j)}) '+formula)
        ax.set_xticks([0,.5,1,1.5,2,2.5]);ax.yaxis.set_major_locator(MaxNLocator(4))
    np.savetxt(ROOT/'数据与代码/物性曲线.csv',np.c_[c,*vals],delimiter=',',header='C_kgkg,rho_kgm3,cp_JkgK,k_WmK,D_1e-9m2s',comments='')
    save(fig,'7-01','物性随含水率变化','按正文第7.2节的附录3经验公式计算。扩散系数固定温度为323.15 K（50℃）；含水率趋于零时扩散系数取极限零。',
         '正文第7.2节物性参数表','水热耦合',{'table':6,'temperature_K':323.15})

def sequential_flow():
    fig,ax=diagram((6.5,2.75),ylim=(0,4.1))
    entries=[('当前状态',r'$C^n,\ T^n$'+'\n读取环境边界',BLUE),('显式更新水分',r'$D(C^n,T^n)$'+'\n'+r'$C^n\rightarrow C^{n+1}$',BLUE),('更新热物性',r'$\rho(C^{n+1}),\ c_p(C^{n+1})$'+'\n'+r'$k(C^{n+1})$',TEAL),('隐式求解温度','向后欧拉\n三对角系统 · Thomas',ORANGE)]
    xs=[.12,2.68,5.24,7.8]
    for j,(x,(title,detail,col)) in enumerate(zip(xs,entries)):
        box(ax,(x,1.5),2.06,1.65,col,'white');txt(ax,x+1.03,2.75,title,col,9.5);txt(ax,x+1.03,2.06,detail,fs=8.5,linespacing=1.7)
        if j<3:arrow(ax,(x+2.06,2.33),(xs[j+1],2.33),GRAY)
    ax.plot([8.83,8.83,1.15],[1.5,.73,.73],c=GRAY,lw=.9);arrow(ax,(1.15,.73),(1.15,1.5),GRAY)
    txt(ax,5,.39,r'$n\leftarrow n+1$，两场每步各更新一次',GRAY,9)
    txt(ax,5,3.7,r'顺序耦合：$N=100$，$\Delta t=0.5\ \mathrm{s}$',fs=10)
    save(fig,'7-02','水热顺序耦合求解流程','问题二径向守恒差分与顺序耦合流程。水分采用显式格式，热物性由更新后的含水率计算，温度采用向后欧拉并由Thomas法求解；步长0.5 s。',
         '正文第7.3节，式(7-11)、(7-12)及求解步骤；第10.3节网格设置','水热耦合')

def center_response():
    fig,axs=plt.subplots(2,1,figsize=(6.5,3.5),sharex=True)
    for ax,data,initial,col,label,panel in [(axs[0],Q2T,28,ORANGE,'中心温度 / °C','(a) 中心热响应'),(axs[1],Q2C,2.55,BLUE,'中心含水率 / (kg/kg)','(b) 中心失水响应')]:
        x=np.r_[0,data[:,0]];y=np.r_[initial,data[:,1]]
        ax.plot(x,y,c=col,marker='o',mfc='white',mew=1);axes_clean(ax,None,label,(-.03,3.03));sublabel(ax,panel);ax.margins(y=.14)
        ax.annotate(f'{y[-1]:.4f}',(x[-1],y[-1]),xytext=(-8,10),textcoords='offset points',ha='right',fontsize=9,color=col)
    axs[1].set_xlabel('时间 / h');axs[1].set_xticks(np.arange(0,3.1,.5))
    save(fig,'7-05','中心温度与含水率响应','采用初始条件以及表3、表4每0.5 h的中心值。折线连接已知时刻；3 h中心温度49.8533℃，中心含水率1.7495 kg/kg。',
         '正文初始条件及表3、表4','水热耦合',{'tables':[7,8],'initial':[28,2.55]})

def drying_curve(code,title,data,positions,table_id,group):
    fig,ax=plt.subplots(figsize=(6.5,3.2));palette=[BLUE,TEAL,PURPLE,ORANGE,GRAY]
    for j,(column,label) in enumerate(positions):
        ax.plot(np.r_[0,data[:,0]],np.r_[2.55,data[:,column]],c=palette[j],marker=['o','s','^','D','v'][j],
                mfc='white',mew=.8,ms=3.8,ls=['-','--','-.',':','-'][j],label=label)
    ax.axhline(.15,c=GRAY,ls='--',lw=.9)
    end=data[-1,0];ax.axvline(end,c=GRAY,ls=':',lw=.8)
    ax.annotate(f'{end:g} h',(end,.15),xytext=(-12,25),textcoords='offset points',ha='right',fontsize=10,color=BLUE,
                arrowprops={'arrowstyle':'-','lw':.8,'color':BLUE})
    axes_clean(ax,'时间 / h','干基含水率 / (kg/kg)',(0,end+2),(.038,3.1));ax.set_yscale('log')
    ax.set_yticks([.05,.15,.5,1,2.55]);ax.set_yticklabels(['0.05','0.15','0.5','1','2.55']);ax.minorticks_off()
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=len(positions),borderaxespad=0,columnspacing=1,fontsize=8.4)
    save(fig,code,title,f'初始含水率与终稿表列时刻的含水率变化。采用对数纵轴；横向虚线为0.15 kg/kg，竖向虚线为表列结束时刻{end:g} h。线段连接表值。',
         f'正文表{5 if table_id==9 else 6}及初始条件',group,{'table':table_id,'columns':[c for c,l in positions],'initial_C':2.55})

def endpoint_grid(code,title,mass):
    fig,ax=diagram((6.5,2.95),ylim=(0,4.4));x0,x1,n=.7,9.3,8;dx=(x1-x0)/n;y0,y1=1.8,2.77
    box(ax,(x0,y0),dx/2,y1-y0,BLUE,'#F0F6FA');box(ax,(x0+3.5*dx,y0),dx,y1-y0,TEAL,'#EAF5F1');box(ax,(x1-dx/2,y0),dx/2,y1-y0,ORANGE,'#FCF3ED')
    for j in range(n):ax.plot([x0+(j+.5)*dx]*2,[y0,y1],c=GRID,lw=.8,zorder=0)
    ax.plot([x0,x1],[2.08]*2,c=INK,lw=.7)
    for j in range(n+1):ax.plot(x0+j*dx,2.08,'o',c=TEAL if j==4 else ORANGE if j==8 else BLUE,ms=3.4)
    for j,label in [(0,'0'),(1,'1'),(3,r'$i-1$'),(4,r'$i$'),(5,r'$i+1$'),(7,r'$N-1$'),(8,r'$N$')]:txt(ax,x0+j*dx,1.48,label,fs=9)
    for edge,label in [(x0+3.5*dx,r'$F_{i-1/2}$'),(x0+4.5*dx,r'$F_{i+1/2}$')]:
        arrow(ax,(edge+.34,2.54),(edge-.34,2.54),TEAL);txt(ax,edge,3.08,label,TEAL,10)
    txt(ax,x0,3.62,r'中心 $m=0$' if mass else r'中心 $\xi=0$',BLUE,9);txt(ax,x0,3.3,'零边界通量',BLUE,8.8)
    txt(ax,x1,3.62,r'表面 $m=M$' if mass else r'表面 $\xi=1$',ORANGE,9);txt(ax,x1,3.3,'对流交换',ORANGE,8.8)
    arrow(ax,(x0+2*dx,.98),(x0+3*dx,.98),GRAY,style='<->')
    txt(ax,x0+2.5*dx,.67,r'$\Delta m=M/N$' if mass else r'$\Delta\xi=1/N$',fs=9)
    txt(ax,7.05,.85,r'端点质量宽度：$\Delta m/2$' if mass else r'$V_i=(\xi_{i+1/2}^2-\xi_{i-1/2}^2)/2$',fs=9.5)
    txt(ax,5,.12,r'端点半控制体；$F$ 向内为正，净流入 $=F_{i+1/2}-F_{i-1/2}$',fs=9)
    section='9.2' if mass else '9.5'
    save(fig,code,title,'状态量位于包含中心和表面的节点，内部为完整控制体，两端为半控制体。界面量F沿用正文向内为正的约定，净流入为右界面量减左界面量。图中节点数量仅作示意。',
         f'正文第{section}节及边界控制体说明','收缩模型')

def radius_compare():
    fig,axs=plt.subplots(2,1,figsize=(6.5,3.7),sharex=True,gridspec_kw={'height_ratios':[2.2,1]})
    t=np.r_[0,BASE[:,0]];r=np.r_[2,BASE[:,1]]
    axs[0].plot(OBS[:,0]/3600,OBS[:,1],c=BLUE,lw=1.3,marker='o',markevery=6,ms=3,mfc='white',label='附件2观测')
    axs[0].plot(t,r,c=ORANGE,ls='--',marker='s',mfc='white',ms=4,label='模型一（正文表值）')
    axs[0].axhline(1.2112,c=GRAY,ls=':',lw=1,label=r'下界 $1.2112$ cm')
    axs[0].annotate('56.2 h',(t[-1],r[-1]),xytext=(-12,34),textcoords='offset points',ha='right',fontsize=9,color=ORANGE,arrowprops={'arrowstyle':'-','lw':.8,'color':ORANGE})
    axes_clean(axs[0],None,r'外半径 $R$ / cm',(-.5,73),(1.15,2.06))
    axs[0].legend(loc='lower left',bbox_to_anchor=(0,1.03),ncol=3,borderaxespad=0,columnspacing=1,fontsize=8.5)
    axs[1].plot(BASE[:,0],BASE[:,3],c=ORANGE,marker='o',mfc='white',ms=4)
    axs[1].axhline(0,c=GRAY,lw=.8);axes_clean(axs[1],'时间 / h','表列偏差 / cm',(-.5,73),(-.01,.27));axs[1].set_xticks([0,12,24,36,48,60,72])
    save(fig,'9-02','模型一半径与观测对照','模型一曲线由初始半径和正文误差分析表中的七个时刻连接而成，止于56.2 h。下图直接展示同表的七个偏差值；理论半径下界为1.2112 cm。',
         '正文第9.3节、第10.1节半径对照表及附件2','收缩模型',{'table':11,'rmse_annotation':False})

def sample_heatmap():
    fig,(ax,ar)=plt.subplots(1,2,figsize=(6.5,4.3),gridspec_kw={'width_ratios':[4.8,1]},sharey=True)
    v=Q4[:,1:5];im=ax.imshow(v,aspect='auto',cmap='viridis',norm=colors.LogNorm(.05,2.55))
    ax.set_xticks(range(4));ax.set_xticklabels(['中心','0.5 cm','1.0 cm',r'表面 $R(t)$']);ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=7)
    ax.set_yticks(range(len(Q4)));ax.set_yticklabels([f'{x:g}' for x in Q4[:,0]]);ax.set_ylabel('时间 / h')
    for i in range(len(Q4)):
        for j in range(4):ax.text(j,i,f'{v[i,j]:.4f}',ha='center',va='center',color='black' if v[i,j]>.65 else 'white',fontsize=9.5)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(Q4),1),minor=True);ax.grid(which='minor',color='white',lw=1.6);ax.tick_params(which='minor',length=0)
    ar.set_xlim(0,1);ar.set_ylim(ax.get_ylim());ar.axis('off');ar.text(.5,1.04,r'$R(t)$ / cm',ha='center',va='bottom',transform=ar.transAxes,fontsize=10)
    for i,value in enumerate(Q4[:,5]):ar.text(.5,i,f'{value:.4f}',ha='center',va='center',fontsize=9.5)
    cb=fig.colorbar(im,ax=ar,fraction=.24,pad=.12);cb.set_ticks([.05,.15,.5,1,2.55]);cb.set_ticklabels(['0.05','0.15','0.5','1','2.55']);cb.minorticks_off();cb.set_label('干基含水率 / (kg/kg)',fontsize=9)
    save(fig,'9-03','收缩模型含水率表值热图','颜色和数字均直接取自表6，色标为对数。各列表示指定采样位置，最右侧含水率列为随外半径移动的药材表面；旁列列出对应半径。矩阵仅展示表列数据，不表示连续空间场。',
         '正文表6','收缩模型',{'table':10,'C':v.tolist(),'radius_cm':Q4[:,5].tolist(),'interpolation':None})

def shrink_flows():
    fig,axs=plt.subplots(2,1,figsize=(6.5,4.1))
    for ax,mass in zip(axs,[True,False]):
        ax.set(xlim=(0,10),ylim=(-.08,2.9));ax.axis('off')
        stages=[('更新几何',r'$C^n\rightarrow r_i^n$' if mass else r'附件2 $\rightarrow R(t)$',TEAL),('显式更新水分',r'$C^n\rightarrow C^{n+1}$',BLUE),('隐式更新温度','热物性更新 · Thomas',ORANGE),('判断是否达标',r'$C_{\mathrm{center}}<0.15$',BLUE)]
        for j,(title,detail,col) in enumerate(stages):
            x=.12+2.56*j;box(ax,(x,.67),2.06,1.05,col,'white');txt(ax,x+1.03,1.41,title,col,9.3);txt(ax,x+1.03,.98,detail,fs=8.3)
            if j<3:arrow(ax,(x+2.06,1.2),(x+2.56,1.2),GRAY)
        ax.plot([8.83,8.83,1.15],[.67,.3,.3],c=GRAY,lw=.8);arrow(ax,(1.15,.3),(1.15,.67),GRAY)
        txt(ax,5,.05,r'未达标：$n\leftarrow n+1$，继续推进',GRAY,8)
        arrow(ax,(8.83,1.72),(8.83,2.08),BLUE)
        txt(ax,8.83,2.26,'达标：记录并结束',BLUE,8.5)
        txt(ax,5,2.72,('(a) 模型一：质量坐标' if mass else '(b) 模型二：随体坐标')+r'，$\Delta t=1\ \mathrm{s}$',fs=10)
    save(fig,'9-06','两种收缩模型求解流程','模型一由当前含水率更新几何位置，模型二由附件2构造外半径。两者均按正文采用水分显式、温度隐式的顺序耦合，步长1 s，并监测中心含水率达标。',
         '正文第9.2、9.5节具体求解步骤','收缩模型')

def sensitivity():
    rows=TABLES[12]['rows'][1:];names=[];pct=[];hours=[];cols=[];current=''
    labels={r'对流传质系数 $h_m$':r'$h_m$',r'扩散系数 $D$':r'$D$',r'对流换热系数 $h$':r'$h$',r'环境温度 $T_a$':r'$T_a$'}
    palette={r'$h_m$':BLUE,r'$D$':PURPLE,r'$h$':TEAL,r'$T_a$':ORANGE}
    for row in rows:
        if row[0]:current=labels[row[0]]
        names.append(current+'  '+row[1]);pct.append(number(row[3]));hours.append(number(row[2]));cols.append(palette[current])
    fig,ax=plt.subplots(figsize=(6.5,4.8));y=np.arange(len(rows))
    ax.barh(y,pct,color=cols,height=.61,alpha=.87);ax.axvline(0,c=GRAY,lw=.8)
    ax.set_yticks(y);ax.set_yticklabels(names);ax.invert_yaxis()
    for i,(value,h) in enumerate(zip(pct,hours)):
        ax.text(value+(.55 if value>=0 else -.55),i,f'{value:+.2f}%  ({h:.2f} h)',ha='left' if value>=0 else 'right',va='center',fontsize=8)
    axes_clean(ax,'烘干时长相对基准变化 / %',None,(-30,37),grid=False)
    ax.grid(axis='x',color=GRID,lw=.5,ls='--');ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    save(fig,'10-01','参数敏感性对比','参数变化、相对基准变化百分比与括号内烘干时长均直接取自正文敏感性表。不同颜色区分四类参数。',
         '正文第10.2节敏感性表','模型检验',{'table':12,'percent':pct,'hours':hours})

def convergence():
    rows=TABLES[13]['rows'][1:];fig,axs=plt.subplots(1,3,figsize=(6.5,2.75))
    for ax,row,col,name in zip(axs,rows,[BLUE,TEAL,ORANGE],['问题三','问题四模型一','问题四模型二']):
        y=np.array([number(row[1]),number(row[2])]);delta=number(row[3]);margin=max(.035,delta*.6)
        ax.plot([100,200],y,c=col,marker='o',mfc='white',mew=1)
        axes_clean(ax,'径向网格数 N','烘干时长 / h' if ax is axs[0] else None,(80,220),(y.min()-margin,y.max()+margin));ax.set_xticks([100,200]);ax.yaxis.set_major_locator(MaxNLocator(4));sublabel(ax,name)
        for x,v in zip([100,200],y):ax.annotate(f'{v:.2f}',(x,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9,color=col)
        ax.text(.5,.08,f'相邻变化 {delta:.2f} h',transform=ax.transAxes,ha='center',fontsize=8.5)
    save(fig,'10-02','网格收敛性对比','直接展示正文中N=100和N=200的烘干时长与相邻变化。为显示数值差异，各子图使用各自的纵轴范围。',
         '正文第10.3节网格收敛表','模型检验',{'table':13})

def main():
    attachment_figures();cylinder();coupling();shrink_map()
    grid_2d();explicit_stencil();end_effects()
    radial_plot('6-01','预热阶段温度径向分布',Q1T,[2,4,6],'温度 / °C','正文表1','预热阶段',4)
    radial_plot('6-02','预热阶段含水率径向分布',Q1C,[2,4,6],'干基含水率 / (kg/kg)','正文表2','预热阶段',5)
    properties();sequential_flow()
    radial_plot('7-03','温度径向分布',Q2T,[0,2,5],'温度 / °C','正文表3','水热耦合',7)
    radial_plot('7-04','含水率径向分布',Q2C,[0,2,5],'干基含水率 / (kg/kg)','正文表4','水热耦合',8)
    center_response()
    drying_curve('8-01','全程含水率与烘干结束时刻',Q3,[(1,'中心'),(2,'0.5 cm'),(3,'1.0 cm'),(4,'1.5 cm'),(5,'表面 2.0 cm')],9,'烘干时长')
    endpoint_grid('9-01','质量坐标节点型控制体网格',True);radius_compare();sample_heatmap()
    radius_input('9-04',6,'半径观测与线性插值','收缩模型')
    endpoint_grid('9-05','随体坐标有限体积网格',False);shrink_flows()
    drying_curve('9-07','收缩模型含水率时间响应',Q4,[(1,'中心'),(2,'0.5 cm'),(3,'1.0 cm'),(4,'表面 R(t)')],10,'收缩模型')
    sensitivity();convergence()
    assert len(RECORDS)==27,len(RECORDS)
    (ROOT/'数据与代码/配图清单.json').write_text(json.dumps(RECORDS,ensure_ascii=False,indent=2))
    print('共完成27幅终稿专用配图。')

if __name__=='__main__':main()
